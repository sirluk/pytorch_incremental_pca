"""Regression test: verify new implementation matches old for float64 inputs.

Ensures that the float64 fix in _incremental_mean_and_var doesn't change
results when the input is already float64 (where no dtype casting occurs).
"""

import torch

from torch_incremental_pca import IncrementalPCA
from torch_incremental_pca.old_incremental_pca import IncrementalPCA as OldIncrementalPCA


def test_float64_regression():
    """New implementation with float64 input should match old implementation exactly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = 200
    n_components = 20
    batch_size = 50
    n_batches = 10

    # Generate deterministic data
    torch.manual_seed(42)
    batches = [torch.randn(batch_size, n_features, device=device, dtype=torch.float64)
               for _ in range(n_batches)]

    old_ipca = OldIncrementalPCA(n_components=n_components, copy=True)
    new_ipca = IncrementalPCA(n_components=n_components, copy=True)

    for batch in batches:
        old_ipca.partial_fit(batch)
        new_ipca.partial_fit(batch)

    # Mean should be identical (both compute in float64, new casts back to input dtype which is float64)
    mean_diff = (old_ipca.mean_ - new_ipca.mean_).abs().max().item()
    print(f"Mean max diff: {mean_diff:.2e}")
    assert mean_diff < 1e-10, f"Mean mismatch: {mean_diff}"

    # Variance
    var_diff = (old_ipca.var_ - new_ipca.var_).abs().max().item()
    print(f"Var max diff: {var_diff:.2e}")
    assert var_diff < 1e-10, f"Variance mismatch: {var_diff}"

    # Components (compare via cosine similarity since sign can flip)
    cos_sim = torch.abs((old_ipca.components_ * new_ipca.components_).sum(dim=1))
    min_cos = cos_sim.min().item()
    print(f"Components min |cosine|: {min_cos:.10f}")
    assert min_cos > 0.9999, f"Component mismatch: min cosine = {min_cos}"

    # Singular values
    sv_diff = (old_ipca.singular_values_ - new_ipca.singular_values_).abs().max().item()
    print(f"Singular values max diff: {sv_diff:.2e}")
    assert sv_diff < 1e-6, f"Singular value mismatch: {sv_diff}"

    # Explained variance
    ev_diff = (old_ipca.explained_variance_ - new_ipca.explained_variance_).abs().max().item()
    print(f"Explained variance max diff: {ev_diff:.2e}")
    assert ev_diff < 1e-6, f"Explained variance mismatch: {ev_diff}"

    print("\nAll regression tests passed!")


if __name__ == "__main__":
    test_float64_regression()
