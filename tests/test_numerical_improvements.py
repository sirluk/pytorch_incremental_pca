"""Numerical regressions and PyTorch integration for the optimized estimator."""

import pytest
import torch

from torch_incremental_pca import IncrementalPCA

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1.0), (torch.float64, 1e8)])
def test_gram_rejects_numerical_null_directions(device, dtype, scale):
    torch.manual_seed(20)
    X = (torch.randn(12, 2, dtype=dtype) @ torch.randn(2, 20, dtype=dtype) * scale).to(
        device
    )

    class CountFallbacks(IncrementalPCA):
        fallbacks = 0

        def _svd_fn_full(self, X):
            self.fallbacks += 1
            return super()._svd_fn_full(X)

    pca = CountFallbacks(n_components=4, gram=True, stats_dtype=dtype).partial_fit(X)
    assert pca.fallbacks == 1
    C = pca.components_
    torch.testing.assert_close(
        C @ C.T, torch.eye(4, device=device, dtype=dtype), atol=2e-5, rtol=2e-5
    )
    reference = torch.linalg.svdvals(X - X.mean(0))[:4]
    torch.testing.assert_close(
        pca.singular_values_, reference, atol=scale * 2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_gram_recovers_well_conditioned_data_without_fallback(device):
    torch.manual_seed(7)
    X = torch.randn(16, 40, device=device)
    pca = IncrementalPCA(n_components=4, gram=True)

    def unexpected_fallback(*args):
        raise AssertionError("Well-conditioned Gram input should use the fast path")

    pca._svd_fn_full = unexpected_fallback
    pca.partial_fit(X)
    C = pca.components_
    torch.testing.assert_close(
        C @ C.T, torch.eye(4, device=device), atol=2e-5, rtol=2e-5
    )
    assert torch.all(pca.singular_values_[:-1] >= pca.singular_values_[1:])


@pytest.mark.parametrize("device", DEVICES)
def test_float64_statistics_preserve_small_variation_on_large_offset(device):
    torch.manual_seed(11)
    X = (1e8 + torch.randn(32, 8, dtype=torch.float64)).to(device)
    pca = IncrementalPCA(n_components=4).fit(X)
    assert pca.var_.dtype == torch.float64
    torch.testing.assert_close(pca.var_, X.var(dim=0, unbiased=False))
    assert pca.explained_variance_ratio_.sum() > 0.5


def test_running_float64_statistics_survive_float32_batches():
    X = torch.full((1000, 1), 1e6 + 1, dtype=torch.float32)
    X[0] = 1e6
    pca = IncrementalPCA(n_components=1, stats_dtype=torch.float64)
    for batch in X.split(1):
        pca.partial_fit(batch)
    assert pca.mean_.dtype == pca.var_.dtype == torch.float64
    torch.testing.assert_close(pca.mean_, X.double().mean(0), rtol=0, atol=1e-8)
    torch.testing.assert_close(
        pca.var_, X.double().var(0, unbiased=False), rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize("device", DEVICES)
def test_transform_stably_centers_large_offset(device):
    torch.manual_seed(42)
    X = (1e6 + torch.randn(64, 64) * 0.25).to(device)
    pca = IncrementalPCA(n_components=8).partial_fit(X)
    reference = (X.double() - pca.mean_.double()) @ pca.components_.double().T
    actual = pca.transform(X)
    assert (actual.double() - reference).norm() / reference.norm() < 2e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "backend", [{}, {"gram": True}, {"lowrank": True, "lowrank_seed": 0}]
)
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_autocast_does_not_change_estimator_precision(backend, autocast_dtype):
    torch.manual_seed(4)
    X = torch.randn(64, 48, device="cuda")
    reference = IncrementalPCA(n_components=8, batch_size=32, **backend).fit(X)
    with torch.autocast("cuda", dtype=autocast_dtype):
        actual = IncrementalPCA(n_components=8, batch_size=32, **backend).fit(X)
        Z = actual.transform(X)
    assert (
        actual.components_.dtype == actual.mean_proj_.dtype == Z.dtype == torch.float32
    )
    torch.testing.assert_close(actual.components_, reference.components_)
    torch.testing.assert_close(Z, reference.transform(X))


@pytest.mark.parametrize("backend", [{}, {"gram": True}, {"lowrank": True}])
def test_retained_tensors_own_only_retained_storage(backend):
    pca = IncrementalPCA(n_components=4, **backend).fit(torch.randn(64, 128))
    for value in (pca.components_, pca.singular_values_):
        assert value.is_contiguous()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_transform_outputs_work_with_trainable_head():
    pca = IncrementalPCA(n_components=3).fit(torch.randn(12, 5))
    Z = pca.transform(torch.randn(4, 5))
    assert not Z.is_inference()
    head = torch.nn.Linear(3, 1)
    head(Z).square().mean().backward()
    assert head.weight.grad is not None


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_precision_context_restores_complete_state_after_exception(precision):
    initial = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision(precision)
        with pytest.raises(RuntimeError, match="sentinel"):
            with IncrementalPCA(allow_tf32=False)._matmul_context():
                assert torch.get_float32_matmul_precision() == "highest"
                raise RuntimeError("sentinel")
        assert torch.get_float32_matmul_precision() == precision
        assert torch.backends.cudnn.allow_tf32 == cudnn
    finally:
        torch.set_float32_matmul_precision(initial)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allow_tf32": False, "matmul_precision": "high"},
        {"allow_tf32": True, "matmul_precision": "highest"},
        {"matmul_precision": "invalid"},
        {"whiten_eps": 0},
        {"whiten_eps": float("nan")},
    ],
)
def test_new_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        IncrementalPCA(**kwargs)


def test_gram_default_batching_keeps_all_inferred_features():
    pca = IncrementalPCA(gram=True).fit(torch.randn(30, 12))
    assert pca.n_components is None
    assert pca.n_components_ == 12
    assert pca.components_.shape == (12, 12)


def test_partial_fit_keeps_initial_compute_dtype():
    pca = IncrementalPCA(n_components=3).partial_fit(
        torch.randn(8, 5, dtype=torch.float64)
    )
    pca.partial_fit(torch.randn(4, 5, dtype=torch.float32))
    assert pca.components_.dtype == torch.float64


def test_raw_gram_helper_works_before_fitting():
    pca = IncrementalPCA(n_components=4, gram=True)
    _, S, Vt, _, _ = pca._svd_fn_gram_topk(torch.randn(16, 32))
    assert S.shape == (4,)
    assert Vt.shape == (4, 32)
