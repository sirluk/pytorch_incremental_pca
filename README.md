# PyTorch Incremental PCA

[![PyPI Version](https://img.shields.io/pypi/v/torch-incremental-pca.svg)](https://pypi.org/project/torch-incremental-pca/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

torch-incremental-pca implements incremental principal component analysis with
PyTorch on CPU and CUDA. It summarizes earlier batches using retained principal
components, then updates those components from each new batch.

## Installation

```bash
pip install torch-incremental-pca
```

PyTorch is the only direct runtime dependency. For development:

```bash
pip install -e ".[dev]"
```

Triton is only used by optional experimental benchmarks; the estimator itself
uses PyTorch.

## Fit tensors, arrays, and memory maps

```python
from torch_incremental_pca import IncrementalPCA

pca = IncrementalPCA(
    n_components=64,
    batch_size=1024,
    compute_device="cuda",
)
pca.fit(X)
```

`X` can be a tensor, a NumPy array or memory map, or a lazy array with `shape`
and row slicing. `fit` reads source metadata first and slices before casting,
making data contiguous, or transferring it. Converted input storage is bounded
by the batch size, in addition to the model and decomposition workspace. A short
final batch may be merged with its predecessor to preserve the existing batching
policy.

`compute_device=None` follows the first input batch's device. Set it explicitly
to fit CPU/disk-backed data on a GPU without loading the full dataset there.
Inputs other than float32/float64 are converted to float32. Later `partial_fit`
batches use the original component dtype and device.

`fit` resets earlier learned state. Inferred settings are recorded in
`batch_size_` and `n_components_`, leaving constructor parameters unchanged.
With automatic batching and `n_components=None`, the inferred rank is
`min(n_samples, n_features)`. With an explicitly smaller batch size, inference
uses the first batch's size.

## Stream training batches

```python
pca = IncrementalPCA(n_components=64, compute_device="cuda")
for X_batch in dataloader:
    pca.partial_fit(X_batch)
```

The first batch must contain at least `n_components` samples. Subsequent batches
may be smaller; feature counts must match. Without an explicit compute device,
a tensor batch on another device is rejected.

`copy=True` preserves inputs. With `copy=False`, the first writable batch may be
centered in place. Read-only arrays are copied after slicing, so read-only memory
maps remain safe. The `check_input` argument is retained for compatibility;
essential shape/device/dtype checks always run.

## Project data with bounded memory

```python
Z = pca.transform(X_new, batch_size=2048, output_device="cpu")
```

Projection centers each batch before multiplying by the component matrix. This
avoids cancellation from subtracting a cached projected mean after the matrix
multiplication. Output dtype matches `components_`; the default output device is
the model device.

A collecting `transform` allocates one complete output. To bound output memory
as well, consume the iterator and release each completed batch:

```python
for Z_batch in pca.transform_batches(X_new, batch_size=2048, output_device="cpu"):
    consume(Z_batch)
```

The iterator returns independent tensors and accepts CPU input for CUDA models.
Do not update the model while consuming an iterator: all batches should use the
same fitted coordinate system.

Alternatively, `out=` accepts a correctly shaped/dtyped tensor or writable NumPy
array, including a NumPy memory map. No full-output concatenation is needed:

```python
pca.transform(X_new, batch_size=2048, out=output_buffer)
```

If both `out` and `output_device` are supplied, their devices must match. A buffer
controls output placement when `output_device` is omitted. Explicitly requesting
batching or setting `compute_device` enables tensor input transfers; ordinary
same-device tensor calls retain the original device checks.

## Reconstruct, whiten, and fit-transform

```python
Z = pca.transform(X_new)
X_reconstructed = pca.inverse_transform(Z)
```

Without whitening, reconstruction is `Z @ components_ + mean_`.
`inverse_transform` supports the same batching, destination, and output-buffer
options; `inverse_transform_batches` streams reconstructed batches.

```python
pca = IncrementalPCA(n_components=64, whiten=True, compute_device="cuda")
Z = pca.fit_transform(X, batch_size=2048, output_device="cpu")
```

Whitening divides scores by the square root of learned explained variance.
Components with standard deviation at most `whiten_eps` times the largest
retained standard deviation are assigned zero scores. Inverse transformation
uses matching scaling. `whiten_eps` defaults to `1e-7` and must be positive.

Incremental truncation makes the learned covariance approximate; whitening does
not promise an exactly identity covariance when replaying all historical data.

`fit_transform` fits first, then makes a second pass using the final basis. It
requires a replayable source and preserves the source during fitting even with
`copy=False`, without copying the full dataset. Scores produced while learning
would otherwise use different coordinate systems.

Fitting and transformation disable autograd and autocast internally. Transform
outputs are ordinary tensors that can feed trainable downstream layers.

## SVD backends

The default uses `torch.linalg.svd`. CUDA driver choice is available through
`svd_driver`; PyTorch's default selects `gesvdj` with a `gesvd` fallback.
`gesvda` is an approximate driver. Performance depends on shape and hardware.

```python
pca = IncrementalPCA(n_components=64, gram=True)
```

Gram mode computes `X @ X.T` for a wide augmented matrix and uses its eigensystem
to recover retained singular vectors. It applies scale-aware diagonal loading
for the eigensolve and recovers singular values from the unshifted data. A
retained eigenvalue near the Gram rounding scale triggers full-SVD fallback,
as do eigensolver failure, nonfinite recovered singular values, and values at
or below `gram_eps`. Tall matrices also use full SVD. This preserves a valid
null-space basis for numerically rank-deficient inputs.

Forming a Gram matrix squares the condition number. Gram mode remains sensitive
to conditioning and reduced-precision matmul; the guard does not make its
well-conditioned numerical accuracy identical to direct SVD. Reported tail
energy excludes the diagonal loading.

```python
pca = IncrementalPCA(
    n_components=64,
    lowrank=True,
    lowrank_q=128,
    lowrank_niter=4,
    lowrank_seed=0,
)
```

Randomized SVD defaults to `q=2*n_components_`, capped by matrix dimensions.
Reducing `q` or iteration count can improve speed while changing weaker
components. Validate approximation quality for your data. Residual noise uses
full discarded energy, including when `lowrank_q == n_components_`.

`gram` and `lowrank` are mutually exclusive. `allow_tf32` and `matmul_precision`
control float32 matmul precision and must agree if both are specified. Their
process-wide setting is restored after use; it is not thread-local. PCA does
not change cuDNN convolution settings.

## Statistics and learned attributes

`stats_dtype` can be float32, float64, or None. The default is float64 on CPU and
the compute dtype on CUDA, preserving float64 CUDA inputs. **Mean and variance
remain in the statistics dtype across updates**, even when components are
float32. Batch statistics are merged using the delta-based Chan update.

The estimator exposes `components_`, `singular_values_`, `mean_`, `var_`,
`explained_variance_`, `explained_variance_ratio_`, `noise_variance_`,
`n_components_`, `n_features_`, `n_samples_seen_`, and `batch_size_` after `fit`.
Retained components and singular values own compact storage. `mean_proj_`
remains available for compatibility; stable transformation does not use it.

Even with direct SVD, incremental results depend on batch size and ordering,
because earlier discarded directions are no longer available to subsequent
updates.

## Testing and benchmarks

```bash
python -m pytest -q
python -m benchmarks.benchmark_backends
```

The comparison benchmark records update times, approximation metrics, retained
storage, and peak additional memory. To compare with a prior implementation:

```bash
git show d962f04:torch_incremental_pca/incremental_pca.py > /tmp/ipca_baseline.py
python -m benchmarks.benchmark_improvements \
    --baseline /tmp/ipca_baseline.py --triton --compile-merge \
    --output /tmp/ipca_results.json
```

`--compile-merge` compares a compiled Chan merge with dynamic sample counts.
`--triton` runs correctness checks for optional CUDA float32 prototypes before
benchmarking them. Compilation and data generation are excluded from timings.
Fitting timings use preloaded GPU input; the separate input-memory measurement
uses a noncontiguous half-precision source. These measurements do not include
disk I/O or CPU-to-GPU transfer throughput.

See [the H100 benchmark report](benchmarks/results/README.md) for measured
tradeoffs, correctness coverage, source hashes, and reproduction commands.

## Acknowledgments

The early implementation adapted code from David Ng's
[PCAonGPU](https://github.com/dnhkng/PCAonGPU), alongside scikit-learn's
IncrementalPCA implementation. This project has since added its own improvements
and extensions. The MIT license and copyright notices for PCAonGPU and this
project are preserved in [LICENSE](LICENSE).
