import pytest
import torch

from torch_incremental_pca import IncrementalPCA


def _subspace_similarity(
    components_a: torch.Tensor, components_b: torch.Tensor
) -> float:
    # 1.0 = identical subspaces
    k = components_a.shape[0]
    overlap = components_a @ components_b.T
    return (torch.linalg.svdvals(overlap).sum() / k).item()


def test_matches_sklearn_incremental_pca_full_backend():
    sklearn = pytest.importorskip("sklearn")
    SklearnIncrementalPCA = sklearn.decomposition.IncrementalPCA

    torch.manual_seed(0)
    n_features = 40
    n_components = 10
    batch_size = 20
    n_batches = 5

    true_components = torch.randn(n_components, n_features, dtype=torch.float64)
    true_components = torch.linalg.qr(true_components.T).Q.T
    strengths = torch.logspace(1, -1, n_components, dtype=torch.float64)

    batches = []
    for _ in range(n_batches):
        coeffs = torch.randn(batch_size, n_components, dtype=torch.float64)
        signal = (coeffs * strengths) @ true_components
        noise = torch.randn(batch_size, n_features, dtype=torch.float64) * 0.1
        batches.append(signal + noise)

    sklearn_ipca = SklearnIncrementalPCA(
        n_components=n_components, batch_size=batch_size
    )
    torch_ipca = IncrementalPCA(n_components=n_components, stats_dtype=torch.float64)

    for batch in batches:
        sklearn_ipca.partial_fit(batch.numpy())
        torch_ipca.partial_fit(batch)

    sklearn_components = torch.from_numpy(sklearn_ipca.components_).to(torch.float64)
    torch_components = torch_ipca.components_.cpu()

    sim = _subspace_similarity(sklearn_components, torch_components)
    assert sim > 0.99

    sklearn_mean = torch.from_numpy(sklearn_ipca.mean_).to(torch.float64)
    mean_diff = (sklearn_mean - torch_ipca.mean_.cpu()).abs().max().item()
    assert mean_diff < 1e-6


def test_matches_sklearn_incremental_pca_gram_backend():
    sklearn = pytest.importorskip("sklearn")
    SklearnIncrementalPCA = sklearn.decomposition.IncrementalPCA

    torch.manual_seed(1)
    n_features = 40
    n_components = 10
    batch_size = 20
    n_batches = 5

    true_components = torch.randn(n_components, n_features, dtype=torch.float64)
    true_components = torch.linalg.qr(true_components.T).Q.T
    strengths = torch.logspace(1, -1, n_components, dtype=torch.float64)

    batches = []
    for _ in range(n_batches):
        coeffs = torch.randn(batch_size, n_components, dtype=torch.float64)
        signal = (coeffs * strengths) @ true_components
        noise = torch.randn(batch_size, n_features, dtype=torch.float64) * 0.1
        batches.append(signal + noise)

    sklearn_ipca = SklearnIncrementalPCA(
        n_components=n_components, batch_size=batch_size
    )
    torch_ipca = IncrementalPCA(
        n_components=n_components, gram=True, stats_dtype=torch.float64
    )

    for batch in batches:
        sklearn_ipca.partial_fit(batch.numpy())
        torch_ipca.partial_fit(batch)

    sklearn_components = torch.from_numpy(sklearn_ipca.components_).to(torch.float64)
    torch_components = torch_ipca.components_.cpu()

    sim = _subspace_similarity(sklearn_components, torch_components)
    assert sim > 0.99
