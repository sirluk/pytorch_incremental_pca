"""Pytest coverage for IncrementalPCA SVD backend consistency."""

from __future__ import annotations

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
