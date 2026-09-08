"""Compare revisions and optional Triton prototypes on one idle CUDA GPU.

Example (from the repository root):
    python -m benchmarks.benchmark_improvements --baseline /tmp/baseline.py
        --triton --compile-merge --output results.json

Timing excludes data generation and compilation. This is an explicit benchmark,
not a test or an automatic backend selector. Triton prototypes support contiguous
CUDA float32 inputs only; the estimator remains pure PyTorch.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from torch_incremental_pca import IncrementalPCA


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("ipca_benchmark_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IncrementalPCA


def make_triton_class():
    from .experimental_kernels import assemble_augmented

    class TritonAugmentationPCA(IncrementalPCA):
        def _build_augmented(self, X, batch_mean, mean_delta, factor):
            tensors = (
                X,
                batch_mean,
                mean_delta,
                self.components_,
                self.singular_values_,
            )
            if not all(
                t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
                for t in tensors
            ):
                return super()._build_augmented(X, batch_mean, mean_delta, factor)
            out = self._get_x_aug_work(
                self.n_components_ + X.shape[0] + 1, X.shape[1], X.device, X.dtype
            )
            return assemble_augmented(
                X,
                batch_mean,
                mean_delta,
                self.components_,
                self.singular_values_,
                factor,
                out,
            )

    return TritonAugmentationPCA


def make_fused_triton_class():
    from .experimental_kernels import prepare_update

    class TritonPreprocessingPCA(IncrementalPCA):
        def _prepare_update(self, X, stats_dtype, first_pass):
            if first_pass or stats_dtype != torch.float32 or not X.is_cuda:
                return super()._prepare_update(X, stats_dtype, first_pass)
            tensors = (
                X,
                self.mean_,
                self.var_,
                self.components_,
                self.singular_values_,
            )
            if not all(t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
                return super()._prepare_update(X, stats_dtype, first_pass)
            batch_var, batch_mean = torch.var_mean(X, dim=0, unbiased=False)
            n1, n2 = self.n_samples_seen_, X.shape[0]
            factor = ((n1 / (n1 + n2)) * n2) ** 0.5
            out = self._get_x_aug_work(
                self.n_components_ + n2 + 1, X.shape[1], X.device, X.dtype
            )
            return prepare_update(
                X,
                batch_mean,
                batch_var,
                self.mean_,
                self.var_,
                self.components_,
                self.singular_values_,
                n1,
                factor,
                out,
            )

    return TritonPreprocessingPCA


def make_compiled_class():
    class CompiledMergePCA(IncrementalPCA):
        _merge_statistics = staticmethod(
            torch.compile(
                IncrementalPCA._merge_statistics, fullgraph=True, dynamic=True
            )
        )

    return CompiledMergePCA


def time_callable(fn, repeats=200):
    for _ in range(10):
        fn()
    wall, events = [], []
    for _ in range(5):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        b.record()
        b.synchronize()
        wall.append((time.perf_counter() - start) * 1e6 / repeats)
        events.append(a.elapsed_time(b) * 1000 / repeats)
    return {"wall_us": statistics.median(wall), "event_us": statistics.median(events)}


def validate_triton():
    from .experimental_kernels import (
        assemble_augmented,
        centered_transform,
        prepare_update,
    )

    errors = []
    for B, D, K in ((17, 37, 5), (64, 256, 16), (500, 2000, 100)):
        for offset in (0, 1e6):
            X = torch.randn(B, D, device="cuda") * 0.25 + offset
            mean = X.mean(0)
            C = torch.linalg.qr(torch.randn(D, K, device="cuda")).Q.T.contiguous()
            inverse_scale = torch.linspace(0, 2, K, device="cuda")
            actual = centered_transform(X, mean, C, inverse_scale)
            reference = (
                (X.double() - mean.double()) @ C.double().T
            ) * inverse_scale.double()
            relative_error = (
                (actual.double() - reference).norm() / reference.norm()
            ).item()
            if relative_error > 2e-5:
                raise AssertionError(
                    f"Triton centered transform error: {relative_error}"
                )
            delta = torch.randn(D, device="cuda")
            S = torch.rand(K, device="cuda")
            factor = 1.234
            out = torch.empty(K + B + 1, D, device="cuda")
            assemble_augmented(X, mean, delta, C, S, factor, out)
            expected = torch.cat((C * S[:, None], X - mean, (delta * -factor)[None]))
            torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
            old_mean = mean - delta
            old_var = torch.rand_like(mean)
            batch_var = X.var(0, unbiased=False)
            prepared = prepare_update(
                X, mean, batch_var, old_mean, old_var, C, S, 500, factor, out
            )
            expected_stats = IncrementalPCA._merge_statistics(
                old_mean, old_var, mean, batch_var, 500, B
            )
            torch.testing.assert_close(
                prepared[1], expected_stats[0], rtol=1e-6, atol=1e-6
            )
            torch.testing.assert_close(
                prepared[2], expected_stats[1], rtol=1e-6, atol=1e-6
            )
            expected[-1] = (mean - old_mean) * -factor
            torch.testing.assert_close(prepared[0], expected, rtol=1e-6, atol=1e-6)
            errors.append(
                {
                    "batch": B,
                    "features": D,
                    "components": K,
                    "offset": offset,
                    "relative_error": relative_error,
                }
            )
    return errors


def fit_timing(cls, kwargs, batches, rounds):
    times, peaks = [], []
    model = None
    for _ in range(rounds):
        model = cls(**kwargs)
        for X in batches[:2]:
            model.partial_fit(X)
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for X in batches[2:]:
            model.partial_fit(X)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000 / (len(batches) - 2))
        peaks.append(torch.cuda.max_memory_allocated() - start_memory)
    C = model.components_
    return model, {
        "update_ms": statistics.median(times),
        "round_update_ms": times,
        "peak_extra_bytes": max(peaks),
        "component_storage_bytes": C.untyped_storage().nbytes(),
        "component_logical_bytes": C.numel() * C.element_size(),
        "max_orthogonality_error": (C @ C.T - torch.eye(C.shape[0], device=C.device))
        .abs()
        .max()
        .item(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--compile-merge", action="store_true")
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required")
    D, K, B = args.features, args.components, args.batch_size
    if not (0 < K <= min(B, D)) or args.updates < 1 or args.rounds < 1:
        parser.error(
            "Require 0 < components <= min(batch-size, features), "
            "positive updates and rounds"
        )
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.set_float32_matmul_precision("highest")
    revisions = {}
    if args.baseline:
        revisions["baseline"] = load_baseline(args.baseline)
    revisions["updated"] = IncrementalPCA
    if args.triton:
        revisions["triton_augmentation"] = make_triton_class()
        revisions["triton_preprocessing"] = make_fused_triton_class()
    if args.compile_merge:
        revisions["compiled_merge"] = make_compiled_class()
    result = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "precision": "highest",
            "cpu_threads": 2,
        },
        "seed": 123,
        "variants": list(revisions),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": {"features": D, "components": K, "batch_size": B},
        "updates_per_round": args.updates,
        "rounds": args.rounds,
        "current_source_sha256": hashlib.sha256(
            Path("torch_incremental_pca/incremental_pca.py").read_bytes()
        ).hexdigest(),
        "fit": [],
        "transform": [],
    }
    if args.baseline:
        result["baseline_source_sha256"] = hashlib.sha256(
            args.baseline.read_bytes()
        ).hexdigest()
    if args.triton:
        import triton

        result["environment"]["triton"] = triton.__version__
        result["experimental_kernels_sha256"] = hashlib.sha256(
            Path(__file__).with_name("experimental_kernels.py").read_bytes()
        ).hexdigest()
        result["triton_validation"] = validate_triton()
        print("Triton correctness checks passed", flush=True)
    # Optional validation must not change the benchmark dataset.
    torch.manual_seed(123)
    C = torch.linalg.qr(torch.randn(D, K, device="cuda")).Q.T
    strengths = torch.logspace(1, -1, K, device="cuda")
    batches = [
        (torch.randn(B, K, device="cuda") * strengths) @ C
        + torch.randn(B, D, device="cuda") * 0.1
        for _ in range(args.updates + 2)
    ]
    models = {}
    for backend, options in (
        ("full", {}),
        ("gram", {"gram": True}),
        ("lowrank", {"lowrank": True, "lowrank_seed": 0}),
    ):
        for name, cls in revisions.items():
            model, metrics = fit_timing(
                cls, {"n_components": K, **options}, batches, args.rounds
            )
            models[backend, name] = model
            reference = models.get((backend, "baseline"), model)
            Qa = torch.linalg.qr(reference.components_.double().T).Q
            Qb = torch.linalg.qr(model.components_.double().T).Q
            cosines = torch.linalg.svdvals(Qa.T @ Qb)
            metrics.update(
                {
                    "backend": backend,
                    "revision": name,
                    "mean_principal_cosine": cosines.mean().item(),
                    "min_principal_cosine": cosines.min().item(),
                    "relative_spectrum_error": (
                        (reference.singular_values_ - model.singular_values_).norm()
                        / reference.singular_values_.norm()
                    ).item(),
                }
            )
            result["fit"].append(metrics)
            print(json.dumps(metrics), flush=True)
    for name in revisions:
        model = models["full", name]

        def fn():
            return model.transform(batches[0])

        metrics = {"revision": name, **time_callable(fn)}
        result["transform"].append(metrics)
        print(json.dumps({"transform": metrics}), flush=True)
    if args.triton:
        from .experimental_kernels import assemble_augmented, centered_transform

        pca = models["full", "updated"]
        X = batches[0]
        mean, delta = X.mean(0), X.mean(0) - pca.mean_
        reference = pca._build_augmented(X, mean, delta, 1.234).clone()
        out = torch.empty_like(reference)
        torch.testing.assert_close(
            assemble_augmented(
                X, mean, delta, pca.components_, pca.singular_values_, 1.234, out
            ),
            reference,
        )
        result["augmentation_microbenchmark"] = {
            "torch": time_callable(lambda: pca._build_augmented(X, mean, delta, 1.234)),
            "triton": time_callable(
                lambda: assemble_augmented(
                    X, mean, delta, pca.components_, pca.singular_values_, 1.234, out
                )
            ),
        }
        result["centered_transform_microbenchmark"] = {
            "torch": time_callable(lambda: (X - pca.mean_) @ pca.components_.T),
            "triton_tf32x3": time_callable(
                lambda: centered_transform(X, pca.mean_, pca.components_)
            ),
        }
        print(
            json.dumps(
                {
                    "microbenchmarks": {
                        key: value
                        for key, value in result.items()
                        if "microbenchmark" in key
                    }
                }
            ),
            flush=True,
        )
    # A noncontiguous half input forces the baseline to materialize whole-dataset
    # contiguous/cast copies. Both estimators use identical update batch sizes.
    source = torch.randn(512, 4096, device="cuda", dtype=torch.float16).T
    result["fit_input_memory"] = []
    for name, cls in revisions.items():
        model = cls(n_components=16, batch_size=128, gram=True)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model.fit(source)
        torch.cuda.synchronize()
        entry = {
            "revision": name,
            "source_shape": list(source.shape),
            "source_dtype": "float16",
            "source_contiguous": False,
            "peak_extra_bytes": torch.cuda.max_memory_allocated() - before,
        }
        result["fit_input_memory"].append(entry)
        print(json.dumps({"fit_input_memory": entry}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
