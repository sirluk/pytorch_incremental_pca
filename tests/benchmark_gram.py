"""Benchmark and numerical comparison of SVD backends for IncrementalPCA.

Compares three backends (full SVD, low-rank SVD, gram eigendecomp) on:
  1. Wall-clock time for partial_fit
  2. Raw SVD-only time (isolating from partial_fit overhead)
  3. Numerical similarity of learned components

Usage:
    python tests/benchmark_gram.py

Requires a CUDA GPU.
"""

import time

import torch

from torch_incremental_pca import IncrementalPCA


def component_similarity(components_a, components_b):
    """Mean absolute cosine similarity between two sets of components."""
    a = components_a / components_a.norm(dim=1, keepdim=True)
    b = components_b / components_b.norm(dim=1, keepdim=True)
    return torch.abs((a * b).sum(dim=1)).mean().item()


def subspace_similarity(components_a, components_b):
    """Subspace overlap via singular values of the projection matrix. 1.0 = identical."""
    k = components_a.shape[0]
    overlap = components_a @ components_b.T
    return (torch.linalg.svdvals(overlap).sum() / k).item()


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
        "lowrank": lambda: IncrementalPCA(n_components=n_components, copy=True, lowrank=True),
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
        ipca_gram._svd_fn_gram(X_test.clone())
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
        ipca_gram._svd_fn_gram(X_test.clone())
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


if __name__ == "__main__":
    main()
