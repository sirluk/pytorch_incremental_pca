"""Benchmark and numerical comparison of SVD backends for IncrementalPCA.

Compares three backends (full SVD, low-rank SVD, gram eigendecomp) on:
  1. Wall-clock time for partial_fit
  2. Raw SVD-only time (isolating from partial_fit overhead)
  3. Numerical similarity of learned components
  4. (Optional) Numerical comparison vs sklearn IncrementalPCA

Usage:
    python tests/test_backends.py

Benchmarking requires a CUDA GPU. The pytest tests run on CPU and optionally CUDA.
"""

from __future__ import annotations

import time

import torch

from torch_incremental_pca import IncrementalPCA


def component_similarity(components_a, components_b):
    """Mean absolute cosine similarity between two sets of components."""
    a = components_a / components_a.norm(dim=1, keepdim=True)
    b = components_b / components_b.norm(dim=1, keepdim=True)
    return torch.abs((a * b).sum(dim=1)).mean().item()


def subspace_similarity(components_a, components_b):
    """Subspace overlap via singular values of the projection matrix.
    1.0 = identical.
    """
    k = components_a.shape[0]
    overlap = components_a @ components_b.T
    return (torch.linalg.svdvals(overlap).sum() / k).item()


def _make_signal_batches(
    *,
    n_features: int,
    n_components: int,
    batch_size: int,
    n_batches: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
):
    torch.manual_seed(seed)
    true_components = torch.randn(n_components, n_features, device=device, dtype=dtype)
    true_components = torch.linalg.qr(true_components.T).Q.T
    strengths = torch.logspace(1, -1, n_components, device=device, dtype=dtype)

    batches = []
    for _ in range(n_batches):
        coeffs = torch.randn(batch_size, n_components, device=device, dtype=dtype)
        signal = (coeffs * strengths) @ true_components
        noise = torch.randn(batch_size, n_features, device=device, dtype=dtype) * 0.1
        batches.append(signal + noise)

    return batches


def _fit_backend(
    *,
    backend: str,
    batches: list[torch.Tensor],
    n_components: int,
    stats_dtype: torch.dtype,
    device: str,
):
    if backend == "full":
        ipca = IncrementalPCA(
            n_components=n_components,
            copy=True,
            stats_dtype=stats_dtype,
        )
    elif backend == "gram":
        ipca = IncrementalPCA(
            n_components=n_components,
            copy=True,
            gram=True,
            stats_dtype=stats_dtype,
        )
    elif backend == "lowrank":
        ipca = IncrementalPCA(
            n_components=n_components,
            copy=True,
            lowrank=True,
            lowrank_seed=0,
            stats_dtype=stats_dtype,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    for batch in batches:
        ipca.partial_fit(batch.to(device=device))
    return ipca


def test_backends_consistent_with_full_cpu():
    n_features = 128
    n_components = 16
    batch_size = 64
    n_batches = 5
    device = "cpu"
    dtype = torch.float64

    batches = _make_signal_batches(
        n_features=n_features,
        n_components=n_components,
        batch_size=batch_size,
        n_batches=n_batches,
        device=device,
        dtype=dtype,
        seed=123,
    )

    full = _fit_backend(
        backend="full",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    gram = _fit_backend(
        backend="gram",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    lowrank = _fit_backend(
        backend="lowrank",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )

    assert gram.components_.shape == full.components_.shape
    assert full.components_.shape == (n_components, n_features)
    assert lowrank.components_.shape == full.components_.shape

    sim_gram = subspace_similarity(full.components_, gram.components_)
    sim_lowrank = subspace_similarity(full.components_, lowrank.components_)

    assert sim_gram > 0.999
    assert sim_lowrank > 0.95


def test_backends_consistent_with_full_cuda():
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_features = 256
    n_components = 32
    batch_size = 128
    n_batches = 5
    device = "cuda"
    dtype = torch.float64

    batches = _make_signal_batches(
        n_features=n_features,
        n_components=n_components,
        batch_size=batch_size,
        n_batches=n_batches,
        device=device,
        dtype=dtype,
        seed=123,
    )

    full = _fit_backend(
        backend="full",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    gram = _fit_backend(
        backend="gram",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    lowrank = _fit_backend(
        backend="lowrank",
        batches=batches,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )

    sim_gram = subspace_similarity(full.components_, gram.components_)
    sim_lowrank = subspace_similarity(full.components_, lowrank.components_)

    assert sim_gram > 0.999
    assert sim_lowrank > 0.95


def test_backends_vs_sklearn_cpu():
    import pytest

    sklearn = pytest.importorskip("sklearn")
    SklearnIncrementalPCA = sklearn.decomposition.IncrementalPCA

    n_features = 128
    n_components = 16
    batch_size = 64
    n_batches = 5
    device = "cpu"
    dtype = torch.float64

    batches_cpu = _make_signal_batches(
        n_features=n_features,
        n_components=n_components,
        batch_size=batch_size,
        n_batches=n_batches,
        device=device,
        dtype=dtype,
        seed=2026,
    )

    sklearn_ipca = SklearnIncrementalPCA(
        n_components=n_components, batch_size=batch_size
    )
    for batch in batches_cpu:
        sklearn_ipca.partial_fit(batch.numpy())

    sklearn_components = torch.from_numpy(sklearn_ipca.components_).to(
        device=device, dtype=dtype
    )

    full = _fit_backend(
        backend="full",
        batches=batches_cpu,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    gram = _fit_backend(
        backend="gram",
        batches=batches_cpu,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    lowrank = _fit_backend(
        backend="lowrank",
        batches=batches_cpu,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )

    sim_full = subspace_similarity(sklearn_components, full.components_)
    sim_gram = subspace_similarity(sklearn_components, gram.components_)
    sim_lowrank = subspace_similarity(sklearn_components, lowrank.components_)

    assert sim_full > 0.99
    assert sim_gram > 0.99
    assert sim_lowrank > 0.90


def test_backends_vs_sklearn_cuda():
    import pytest

    sklearn = pytest.importorskip("sklearn")
    SklearnIncrementalPCA = sklearn.decomposition.IncrementalPCA

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n_features = 256
    n_components = 32
    batch_size = 128
    n_batches = 5
    device = "cuda"
    dtype = torch.float64

    batches_cpu = _make_signal_batches(
        n_features=n_features,
        n_components=n_components,
        batch_size=batch_size,
        n_batches=n_batches,
        device="cpu",
        dtype=dtype,
        seed=2026,
    )

    sklearn_ipca = SklearnIncrementalPCA(
        n_components=n_components, batch_size=batch_size
    )
    for batch in batches_cpu:
        sklearn_ipca.partial_fit(batch.numpy())

    sklearn_components = torch.from_numpy(sklearn_ipca.components_).to(
        device=device, dtype=dtype
    )

    batches_cuda = [b.to(device) for b in batches_cpu]

    full = _fit_backend(
        backend="full",
        batches=batches_cuda,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    gram = _fit_backend(
        backend="gram",
        batches=batches_cuda,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )
    lowrank = _fit_backend(
        backend="lowrank",
        batches=batches_cuda,
        n_components=n_components,
        stats_dtype=dtype,
        device=device,
    )

    sim_full = subspace_similarity(sklearn_components, full.components_)
    sim_gram = subspace_similarity(sklearn_components, gram.components_)
    sim_lowrank = subspace_similarity(sklearn_components, lowrank.components_)

    assert sim_full > 0.99
    assert sim_gram > 0.99
    assert sim_lowrank > 0.90


def compare_with_sklearn(device: str):
    try:
        from sklearn.decomposition import IncrementalPCA as SklearnIncrementalPCA
    except Exception as exc:
        print(f"\n=== sklearn comparison (skipped: {exc}) ===")
        return

    # Keep this fairly small: sklearn runs on CPU and can be slow.
    n_features = 256
    n_components = 32
    batch_size = 128
    n_reps = 20
    dtype = torch.float64

    torch.manual_seed(2026)
    true_components = torch.randn(n_components, n_features, dtype=dtype)
    true_components = torch.linalg.qr(true_components.T).Q.T
    strengths = torch.logspace(1, -1, n_components, dtype=dtype)

    torch.manual_seed(123)
    batches_cpu = []
    for _ in range(n_reps):
        coeffs = torch.randn(batch_size, n_components, dtype=dtype)
        signal = (coeffs * strengths) @ true_components
        noise = torch.randn(batch_size, n_features, dtype=dtype) * 0.1
        batches_cpu.append(signal + noise)

    sklearn_ipca = SklearnIncrementalPCA(
        n_components=n_components, batch_size=batch_size
    )
    for batch in batches_cpu:
        sklearn_ipca.partial_fit(batch.numpy())

    # Match sklearn's float64 stats for numerical comparability.
    methods = {
        "full": lambda: IncrementalPCA(
            n_components=n_components, copy=True, stats_dtype=dtype
        ),
        "gram": lambda: IncrementalPCA(
            n_components=n_components, copy=True, gram=True, stats_dtype=dtype
        ),
        "lowrank": lambda: IncrementalPCA(
            n_components=n_components,
            copy=True,
            lowrank=True,
            lowrank_seed=0,
            stats_dtype=dtype,
        ),
    }

    gpu_results = {}
    for name, make_ipca in methods.items():
        ipca = make_ipca()
        for batch in batches_cpu:
            ipca.partial_fit(batch.to(device))
        gpu_results[name] = ipca

    sklearn_components = torch.from_numpy(sklearn_ipca.components_).to(
        device=device, dtype=dtype
    )
    print("\n=== sklearn comparison (numerical) ===")
    for name in ["full", "gram", "lowrank"]:
        sim = component_similarity(sklearn_components, gpu_results[name].components_)
        sub = subspace_similarity(sklearn_components, gpu_results[name].components_)
        print(f"  {name:>8}: cosine={sim:.6f}, subspace={sub:.6f}")


def main():
    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU."
    device = "cuda"

    n_features = 2000
    n_components = 100
    batch_size = 500
    n_reps = 100

    print(f"Benchmark: D={n_features}, K={n_components}, batch_size={batch_size}")
    m = n_components + batch_size + 1
    print(f"Augmented matrix shape: ({m}, {n_features})")
    print()

    # Create true components and synthetic data
    torch.manual_seed(42)
    true_components = torch.randn(n_components, n_features, device=device)
    true_components = torch.linalg.qr(true_components.T).Q.T
    strengths = torch.logspace(1, -1, n_components, device=device)

    # Pre-generate all batches (not timed)
    torch.manual_seed(123)
    batches = []
    for _ in range(n_reps):
        coeffs = torch.randn(batch_size, n_components, device=device)
        signal = (coeffs * strengths) @ true_components
        noise = torch.randn(batch_size, n_features, device=device) * 0.1
        batches.append(signal + noise)

    # --- partial_fit benchmark ---
    print("=== partial_fit benchmark ===")
    methods = {
        "full": lambda: IncrementalPCA(n_components=n_components, copy=True),
        "gram": lambda: IncrementalPCA(n_components=n_components, copy=True, gram=True),
        "lowrank": lambda: IncrementalPCA(
            n_components=n_components, copy=True, lowrank=True
        ),
    }

    results = {}
    for name, make_ipca in methods.items():
        # Warmup (3 batches, not timed)
        ipca_warmup = make_ipca()
        for i in range(3):
            ipca_warmup.partial_fit(batches[i])
        torch.cuda.synchronize()

        # Timed run
        ipca = make_ipca()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in batches:
            ipca.partial_fit(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        results[name] = {"model": ipca, "time": elapsed}
        print(f"{name:>8}: {elapsed:.3f}s")

    print()
    full_time = results["full"]["time"]
    for name in ["gram", "lowrank"]:
        speedup = full_time / results[name]["time"]
        print(f"{name:>8} speedup vs full: {speedup:.2f}x")

    # --- Raw SVD-only benchmark ---
    print("\n=== Raw SVD-only benchmark ===")
    ipca_full = IncrementalPCA(n_components=n_components)
    ipca_gram = IncrementalPCA(n_components=n_components, gram=True)
    X_test = torch.randn(m, n_features, device=device)

    # Warmup
    for _ in range(5):
        ipca_full._svd_fn_full(X_test.clone())
        ipca_gram._svd_fn_gram_topk(X_test.clone())
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        ipca_full._svd_fn_full(X_test.clone())
    torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        ipca_gram._svd_fn_gram_topk(X_test.clone())
    torch.cuda.synchronize()
    t_gram = time.perf_counter() - t0

    print(f"    full SVD: {t_full:.3f}s ({n_reps} calls)")
    print(f"  gram eigh: {t_gram:.3f}s ({n_reps} calls)")
    print(f"  gram speedup: {t_full / t_gram:.2f}x")

    # --- Numerical comparison ---
    print("\n=== Numerical similarity ===")
    ref = results["full"]["model"].components_
    print("vs full SVD (mean |cosine|):")
    for name in ["gram", "lowrank"]:
        sim = component_similarity(ref, results[name]["model"].components_)
        sub = subspace_similarity(ref, results[name]["model"].components_)
        print(f"  {name:>8}: cosine={sim:.6f}, subspace={sub:.6f}")

    print("vs true components:")
    for name in ["full", "gram", "lowrank"]:
        sim = component_similarity(true_components, results[name]["model"].components_)
        sub = subspace_similarity(true_components, results[name]["model"].components_)
        print(f"  {name:>8}: cosine={sim:.6f}, subspace={sub:.6f}")

    compare_with_sklearn(device)


if __name__ == "__main__":
    main()
