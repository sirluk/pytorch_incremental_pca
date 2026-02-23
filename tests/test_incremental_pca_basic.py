import pytest
import torch

from torch_incremental_pca import IncrementalPCA


def test_fit_raises_when_batch_size_less_than_n_components():
    X = torch.randn(20, 10)
    ipca = IncrementalPCA(n_components=5, batch_size=4)
    with pytest.raises(ValueError, match=r"batch_size=.*n_components"):
        ipca.fit(X)


def test_fit_gram_auto_batch_size_produces_expected_shapes():
    # Regression for gram=True auto-batch_size picking too-small batches.
    X = torch.randn(30, 12)
    ipca = IncrementalPCA(n_components=10, gram=True)
    ipca.fit(X)

    assert ipca.components_.shape == (10, 12)
    assert ipca.singular_values_.shape == (10,)


def test_copy_flag_controls_input_mutation_on_first_partial_fit():
    X = torch.randn(20, 10)
    X_orig = X.clone()
    ipca = IncrementalPCA(n_components=5, copy=True)
    ipca.partial_fit(X)
    assert torch.allclose(X, X_orig)

    X2 = torch.randn(20, 10)
    X2_orig = X2.clone()
    ipca2 = IncrementalPCA(n_components=5, copy=False)
    ipca2.partial_fit(X2)

    assert not torch.allclose(X2, X2_orig)
    assert torch.allclose(
        ipca2.mean_, X2_orig.mean(dim=0), atol=1e-5, rtol=0
    ), "mean_ should be computed from the original (pre-mutation) batch"
    assert torch.allclose(
        X2.mean(dim=0), torch.zeros(10), atol=1e-5, rtol=0
    ), "copy=False should center the batch in-place on the first pass"


def test_partial_fit_casts_non_float32_64_to_float32():
    X16 = torch.randn(20, 10).to(torch.float16)
    ipca = IncrementalPCA(n_components=5)
    ipca.partial_fit(X16)
    assert ipca.mean_.dtype == torch.float32
    assert ipca.components_.dtype == torch.float32


def test_lowrank_backend_runs_and_produces_expected_shapes():
    X = torch.randn(25, 15)
    ipca = IncrementalPCA(n_components=5, lowrank=True, lowrank_seed=0)
    ipca.partial_fit(X)
    assert ipca.components_.shape == (5, 15)
    assert ipca.singular_values_.shape == (5,)


def test_transform_casts_to_components_dtype():
    X_fit = torch.randn(30, 10, dtype=torch.float64)
    ipca = IncrementalPCA(n_components=5)
    ipca.partial_fit(X_fit)

    X = torch.randn(7, 10, dtype=torch.float32)
    Y = ipca.transform(X)

    assert Y.dtype == torch.float64

    X64 = X.to(torch.float64)
    manual = (X64 @ ipca.components_.T) - ipca.mean_proj_
    assert torch.allclose(Y, manual)


def test_transform_raises_on_device_mismatch():
    X_fit = torch.randn(30, 10)
    ipca = IncrementalPCA(n_components=5)
    ipca.partial_fit(X_fit)

    try:
        X_meta = torch.empty(2, 10, device="meta", dtype=ipca.components_.dtype)
    except Exception:
        pytest.skip("meta device not supported in this torch build")

    with pytest.raises(ValueError, match=r"device"):
        ipca.transform(X_meta)


def test_zero_variance_batch_produces_finite_ratios():
    X = torch.ones(10, 5)
    ipca = IncrementalPCA(n_components=3)
    ipca.partial_fit(X)

    assert torch.isfinite(ipca.explained_variance_ratio_).all()
    assert torch.all(ipca.explained_variance_ratio_ == 0)
    assert torch.isfinite(ipca.noise_variance_).item()


def test_single_sample_batch_produces_finite_stats():
    X = torch.randn(1, 5)
    ipca = IncrementalPCA()
    ipca.partial_fit(X)

    assert ipca.n_components == 1
    assert torch.isfinite(ipca.explained_variance_).all()
    assert torch.isfinite(ipca.explained_variance_ratio_).all()
    assert torch.isfinite(ipca.noise_variance_).item()
