"""Streaming contracts and reconstruction/whitening behavior."""

import numpy as np
import pytest
import torch

from torch_incremental_pca import IncrementalPCA

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class SliceOnlySource:
    """A lazy array that refuses whole-array conversion and oversized reads."""

    def __init__(self, data, max_rows):
        self.data = data
        self.shape = data.shape
        self.max_rows = max_rows
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("Attempted to materialize the whole source")

    def __getitem__(self, index):
        assert isinstance(index, slice)
        rows = len(range(*index.indices(self.shape[0])))
        assert rows <= self.max_rows
        self.reads.append(index)
        return self.data[index]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("check_input", [True, False])
def test_fit_slices_before_casting_and_transferring(device, check_input):
    data = np.arange(40 * 12, dtype=np.int16).reshape(40, 12)[:, ::2]
    source = SliceOnlySource(data, max_rows=8)
    pca = IncrementalPCA(n_components=3, batch_size=8, compute_device=device).fit(
        source, check_input=check_input
    )
    assert len(source.reads) == 5
    assert pca.components_.device.type == device
    assert pca.components_.dtype == torch.float32
    torch.testing.assert_close(
        pca.mean_.cpu().double(), torch.from_numpy(data.astype(np.float64)).mean(0)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_readonly_memmap_is_not_mutated_with_copy_false(tmp_path, device):
    path = tmp_path / "input.npy"
    data = np.random.default_rng(5).normal(size=(24, 8)).astype(np.float32)
    np.save(path, data)
    mapped = np.load(path, mmap_mode="r")
    pca = IncrementalPCA(
        n_components=3, batch_size=8, compute_device=device, copy=False
    ).fit(mapped)
    np.testing.assert_array_equal(np.load(path), data)
    assert pca.n_samples_seen_ == 24


@pytest.mark.parametrize("device", DEVICES)
def test_transform_iterator_is_lazy_and_does_not_reuse_outputs(device):
    torch.manual_seed(17)
    X = torch.randn(24, 8)
    pca = IncrementalPCA(n_components=3, compute_device=device).fit(X)
    source = SliceOnlySource(X.numpy(), max_rows=5)
    batches = pca.transform_batches(source, batch_size=5, output_device="cpu")
    assert source.reads == []
    first = next(batches)
    saved = first.clone()
    assert len(source.reads) == 1
    assert first.device.type == "cpu"
    assert torch.is_grad_enabled()
    tail = list(batches)
    torch.testing.assert_close(first, saved)
    result = torch.cat([first, *tail])
    reference = pca.transform(X, batch_size=24, output_device="cpu")
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_output_buffers_include_noncontiguous_tensors_and_memmaps(tmp_path, device):
    torch.manual_seed(18)
    X = torch.randn(24, 8)
    pca = IncrementalPCA(n_components=3, compute_device=device).fit(X)
    reference = pca.transform(X, batch_size=24, output_device="cpu")
    tensor_out = torch.empty(3, 24).T
    result = pca.transform(X, batch_size=5, out=tensor_out)
    assert result is tensor_out
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)
    mmap_out = np.memmap(
        tmp_path / "scores.dat", mode="w+", dtype="float32", shape=(24, 3)
    )
    assert pca.transform(X, batch_size=5, out=mmap_out) is mmap_out
    np.testing.assert_allclose(mmap_out, reference.numpy(), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("whiten", [False, True])
def test_whitened_and_unwhitened_reconstruction(device, whiten):
    torch.manual_seed(19)
    X = torch.randn(40, 8, dtype=torch.float64)
    pca = IncrementalPCA(n_components=8, compute_device=device, whiten=whiten).fit(X)
    Z = pca.transform(X, batch_size=7)
    restored = pca.inverse_transform(Z, batch_size=9, output_device="cpu")
    torch.testing.assert_close(restored, X, rtol=1e-8, atol=1e-8)
    if whiten:
        covariance = (Z - Z.mean(0)).T @ (Z - Z.mean(0)) / (X.shape[0] - 1)
        torch.testing.assert_close(
            covariance.cpu(), torch.eye(8, dtype=torch.float64), rtol=1e-8, atol=1e-8
        )
    chunks = list(pca.inverse_transform_batches(Z, batch_size=7, output_device="cpu"))
    torch.testing.assert_close(torch.cat(chunks), restored)


@pytest.mark.parametrize("device", DEVICES)
def test_whitening_degenerate_directions_are_zero(device):
    X = torch.ones(12, 8, device=device)
    pca = IncrementalPCA(n_components=4, whiten=True).fit(X)
    Z = pca.transform(torch.randn_like(X))
    assert torch.isfinite(Z).all()
    assert torch.count_nonzero(Z) == 0
    torch.testing.assert_close(pca.inverse_transform(Z), X)


def test_fit_transform_replays_original_data_with_final_basis():
    torch.manual_seed(21)
    X = (
        torch.randn(30, 8, dtype=torch.float64)
        + torch.arange(30, dtype=torch.float64)[:, None]
    )
    original = X.clone()
    pca = IncrementalPCA(n_components=3, batch_size=10, copy=False)
    actual = pca.fit_transform(X, batch_size=7)
    assert pca.copy is False
    torch.testing.assert_close(X, original)
    expected = (original - pca.mean_) @ pca.components_.T
    torch.testing.assert_close(actual, expected)


def test_fit_transform_restores_copy_parameter_after_failure():
    pca = IncrementalPCA(n_components=4, copy=False)
    with pytest.raises(ValueError):
        pca.fit_transform(torch.randn(2, 6))
    assert pca.copy is False


@pytest.mark.parametrize("inverse", [False, True])
def test_empty_projection_preserves_shape_and_dtype(inverse):
    pca = IncrementalPCA(n_components=3).fit(torch.randn(10, 8))
    X = torch.empty(0, 3 if inverse else 8)
    result = (pca.inverse_transform if inverse else pca.transform)(X)
    assert result.shape == (0, 8 if inverse else 3)
    assert list(pca.transform_batches(torch.empty(0, 8))) == []


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, True])
def test_projection_rejects_invalid_batch_sizes(batch_size):
    pca = IncrementalPCA(n_components=3).fit(torch.randn(10, 8))
    with pytest.raises(ValueError, match="batch_size"):
        pca.transform(torch.randn(4, 8), batch_size=batch_size)


def test_output_validation(tmp_path):
    pca = IncrementalPCA(n_components=3).fit(torch.randn(10, 8))
    X = torch.randn(4, 8)
    for out in (torch.empty(4, 4), torch.empty(4, 3, dtype=torch.float64)):
        with pytest.raises(ValueError, match="out"):
            pca.transform(X, out=out)
    mapped = np.memmap(tmp_path / "out.dat", mode="w+", shape=(4, 3), dtype="float32")
    mapped.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        pca.transform(X, out=mapped)


@pytest.mark.parametrize("device", DEVICES)
def test_output_device_aliases_match_explicit_buffer_device(device):
    pca = IncrementalPCA(n_components=3, compute_device=device).fit(torch.randn(12, 8))
    X = torch.randn(4, 8)
    out = torch.empty(4, 3, device=device)
    alias = "cpu:0" if device == "cpu" else "cuda"
    assert pca.transform(X, output_device=alias, out=out) is out
    torch.testing.assert_close(out, pca.transform(X))
