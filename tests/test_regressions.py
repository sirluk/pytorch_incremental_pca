import math

import pytest
import torch

from torch_incremental_pca import IncrementalPCA

LEARNED_ATTRIBUTES = (
    "components_",
    "singular_values_",
    "mean_",
    "var_",
    "explained_variance_",
    "explained_variance_ratio_",
    "noise_variance_",
    "mean_proj_",
    "n_components_",
    "batch_size_",
    "n_features_",
    "n_samples_seen_",
)


def _assert_orthonormal_rows(components: torch.Tensor, atol: float = 1e-8):
    identity = torch.eye(
        components.shape[0], dtype=components.dtype, device=components.device
    )
    torch.testing.assert_close(
        components @ components.mT, identity, rtol=atol, atol=atol
    )


def test_repeated_fit_restarts_and_matches_fresh_estimator():
    torch.manual_seed(10)
    first = torch.randn(21, 6, dtype=torch.float64)
    second = torch.randn(21, 9, dtype=torch.float64)

    estimator = IncrementalPCA(n_components=4, batch_size=7)
    estimator.fit(first)
    assert estimator._x_aug_work is not None
    assert estimator.n_features_ == 6

    estimator.fit(second)
    fresh = IncrementalPCA(n_components=4, batch_size=7).fit(second)

    assert estimator.n_samples_seen_ == second.shape[0]
    assert estimator.n_features_ == second.shape[1]
    assert estimator._x_aug_work.shape[1] == second.shape[1]
    torch.testing.assert_close(estimator.mean_, fresh.mean_)
    torch.testing.assert_close(estimator.singular_values_, fresh.singular_values_)
    torch.testing.assert_close(estimator.components_, fresh.components_)


def test_fit_clears_learned_state_before_validation():
    estimator = IncrementalPCA(n_components=2, batch_size=4)
    estimator.fit(torch.randn(8, 5))
    assert estimator._x_aug_work is not None

    with pytest.raises(ValueError, match="nonempty"):
        estimator.fit(torch.empty(0, 5))

    for attribute in LEARNED_ATTRIBUTES:
        assert not hasattr(estimator, attribute)
    assert estimator._x_aug_work is None


def test_inferred_values_use_learned_attributes_without_mutating_parameters():
    estimator = IncrementalPCA()
    estimator.fit(torch.randn(8, 5))

    assert estimator.n_components is None
    assert estimator.batch_size is None
    assert estimator.n_components_ == 5
    assert estimator.batch_size_ == 25


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_components": -1}, "n_components"),
        ({"n_components": 1.5}, "n_components"),
        ({"n_components": True}, "n_components"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": -1}, "batch_size"),
        ({"batch_size": 2.5}, "batch_size"),
        ({"lowrank_niter": -1}, "lowrank_niter"),
        ({"lowrank_niter": 1.5}, "lowrank_niter"),
        ({"gram_eps": 0.0}, "gram_eps"),
        ({"gram_eps": -1.0}, "gram_eps"),
        ({"gram_eps": math.inf}, "gram_eps"),
        ({"gram_eps": math.nan}, "gram_eps"),
        ({"stats_dtype": torch.float16}, "stats_dtype"),
        ({"stats_dtype": torch.int64}, "stats_dtype"),
    ],
)
def test_parameter_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        IncrementalPCA(**kwargs)


@pytest.mark.parametrize("method_name", ["fit", "partial_fit"])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (4,),
        (2, 3, 4),
        (0, 3),
        (3, 0),
    ],
)
def test_fit_inputs_must_be_nonempty_and_2d(method_name, shape):
    X = torch.empty(shape)
    estimator = IncrementalPCA()

    with pytest.raises(ValueError, match="2D|nonempty"):
        getattr(estimator, method_name)(X)


def test_only_first_partial_fit_requires_enough_samples():
    estimator = IncrementalPCA(n_components=4)
    with pytest.raises(ValueError, match="first batch"):
        estimator.partial_fit(torch.randn(3, 6))

    estimator.partial_fit(torch.randn(8, 6))
    estimator.partial_fit(torch.randn(2, 6))

    assert estimator.n_samples_seen_ == 10
    assert estimator.components_.shape == (4, 6)


@pytest.mark.parametrize(
    ("kind", "scale"),
    [
        ("well_conditioned", 1.0),
        ("well_conditioned", 1e-5),
        ("well_conditioned", 1e8),
        ("zero", 1.0),
        ("rank_deficient", 1.0),
    ],
)
def test_gram_matches_full_svd_across_rank_and_scale(kind, scale):
    torch.manual_seed(20)
    if kind == "well_conditioned":
        X = torch.randn(12, 20, dtype=torch.float64) * scale
    elif kind == "zero":
        X = torch.zeros(12, 20, dtype=torch.float64)
    else:
        X = torch.randn(12, 2, dtype=torch.float64) @ torch.randn(
            2, 20, dtype=torch.float64
        )

    full = IncrementalPCA(n_components=4, stats_dtype=torch.float64)
    gram = IncrementalPCA(
        n_components=4, gram=True, stats_dtype=torch.float64, gram_eps=1e-7
    )
    full.partial_fit(X)
    gram.partial_fit(X)

    assert torch.isfinite(gram.components_).all()
    assert torch.isfinite(gram.singular_values_).all()
    assert torch.isfinite(gram.noise_variance_)
    _assert_orthonormal_rows(gram.components_)
    torch.testing.assert_close(
        gram.singular_values_,
        full.singular_values_,
        rtol=2e-6,
        atol=max(1.0, scale) * 1e-10,
    )
    torch.testing.assert_close(
        gram.components_.mT @ gram.components_,
        full.components_.mT @ full.components_,
        rtol=2e-6,
        atol=2e-7,
    )

    if kind in {"zero", "rank_deficient"}:
        assert gram.noise_variance_.abs().item() < 1e-20


def test_gram_tail_energy_has_no_ridge_variance():
    X = torch.zeros(4, 8, dtype=torch.float64)
    X[:, 0] = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    X[:, 1] = torch.tensor([-1.0, -1.0, 1.0, 1.0])

    estimator = IncrementalPCA(n_components=2, gram=True, stats_dtype=torch.float64)
    estimator.partial_fit(X)

    torch.testing.assert_close(
        estimator.singular_values_,
        torch.tensor([2.0, 2.0], dtype=torch.float64),
    )
    assert estimator.noise_variance_.item() < 1e-15
    _assert_orthonormal_rows(estimator.components_)


def test_gram_eigh_failure_falls_back_to_full_svd(monkeypatch):
    X = torch.randn(10, 16, dtype=torch.float64)
    expected = IncrementalPCA(n_components=4, stats_dtype=torch.float64)
    expected.partial_fit(X)

    def fail_eigh(*_args, **_kwargs):
        raise torch.linalg.LinAlgError("forced eigensolver failure")

    monkeypatch.setattr(torch.linalg, "eigh", fail_eigh)
    actual = IncrementalPCA(n_components=4, gram=True, stats_dtype=torch.float64)
    actual.partial_fit(X)

    torch.testing.assert_close(actual.singular_values_, expected.singular_values_)
    torch.testing.assert_close(actual.components_, expected.components_)


def test_lowrank_noise_uses_full_residual_energy_when_q_equals_components():
    torch.manual_seed(30)
    X = torch.randn(18, 12, dtype=torch.float64)
    estimator = IncrementalPCA(
        n_components=4,
        lowrank=True,
        lowrank_q=4,
        lowrank_niter=3,
        lowrank_seed=0,
        stats_dtype=torch.float64,
    )
    estimator.partial_fit(X)

    centered = X - X.mean(dim=0)
    discarded_count = min(centered.shape) - estimator.n_components_
    expected = (
        centered.square().sum() - estimator.singular_values_.square().sum()
    ).clamp(min=0) / ((X.shape[0] - 1) * discarded_count)

    assert torch.isfinite(estimator.noise_variance_)
    assert estimator.noise_variance_.item() > 0
    torch.testing.assert_close(estimator.noise_variance_, expected)
