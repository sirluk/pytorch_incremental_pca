# PyTorch Incremental PCA

[![PyPI Version](https://img.shields.io/pypi/v/torch-incremental-pca.svg)](https://pypi.org/project/torch-incremental-pca/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

torch-incremental-pca is a PyTorch implementation of incremental principal
component analysis. It follows the numerical update used by scikit-learn's
IncrementalPCA, works with CPU and CUDA tensors, and processes datasets in
batches so the full input does not need to fit in memory.

## Installation

~~~bash
pip install torch-incremental-pca
~~~

PyTorch is the only direct runtime dependency. Development and comparison
dependencies can be installed from a source checkout with:

~~~bash
pip install -e ".[dev]"
~~~

## Fit a tensor

fit() divides a nonempty 2D tensor into batches and resets any model state
learned by an earlier fit() call.

~~~python
import torch

from torch_incremental_pca import IncrementalPCA

X = torch.randn(100_000, 512, device="cuda")

pca = IncrementalPCA(n_components=64, batch_size=2048)
pca.fit(X)

print(pca.components_.shape)  # torch.Size([64, 512])
~~~

When batch_size=None, fit() records an inferred size in batch_size_ without
changing the constructor parameter. The usual inferred size is five times the
feature count; Gram mode may choose a smaller, wide-matrix-friendly batch.
Likewise, an inferred component count is stored in n_components_ while
n_components remains None.

## Stream batches

Call partial_fit() when data already arrives in batches:

~~~python
pca = IncrementalPCA(n_components=64)

for X_batch in dataloader:
    pca.partial_fit(X_batch.to("cuda"))
~~~

The first batch must contain at least n_components samples. Later batches may
be smaller, but every batch must have the same number of features and be on the
same device as the fitted model. Inputs are converted to float32 unless they
are already float32 or float64.

With copy=False, the first tensor passed to partial_fit() may be centered in
place. The default, copy=True, preserves the input.

## Transform data

~~~python
Z = pca.transform(X_new.to("cuda"))
~~~

transform() returns (X_new - mean_) @ components_.T. It accepts input with a
different floating-point dtype and casts it to the component dtype, but a
tensor on a different device is rejected. The mean projection is cached, so
the implementation does not need to allocate a centered copy during
transformation.

## SVD backends

The default backend uses torch.linalg.svd and is the reference exact
implementation.

~~~python
pca = IncrementalPCA(n_components=64)
~~~

Gram mode is intended for wide augmented matrices, especially on GPUs where
matrix multiplication and symmetric eigendecomposition can be faster than a
full SVD:

~~~python
pca = IncrementalPCA(n_components=64, gram=True)
~~~

It forms X @ X.T, applies scale-aware diagonal loading only while solving the
eigenproblem, and recovers singular values from the original unshifted matrix.
It falls back to full SVD when the matrix is tall, when torch.linalg.eigh
fails, or when a retained singular value is nonfinite or no larger than
gram_eps. This fallback gives rank-deficient inputs a valid orthonormal
null-space basis. gram_eps is an absolute singular-value safety threshold and
is not added to reported singular values or variances.

Low-rank mode uses PyTorch's randomized torch.svd_lowrank:

~~~python
pca = IncrementalPCA(
    n_components=64,
    lowrank=True,
    lowrank_q=128,
    lowrank_niter=4,
    lowrank_seed=0,
)
~~~

If lowrank_q=None, the effective rank defaults to twice n_components_ and is
capped by the matrix dimensions. Increasing lowrank_q or lowrank_niter
generally improves the approximation at additional cost. Set lowrank_seed for
repeatable randomized decompositions.

gram=True and lowrank=True are mutually exclusive. CUDA full-SVD behavior can
also be tuned with svd_driver; matrix multiplication can be controlled with
allow_tf32 and matmul_precision.

## Statistics and learned attributes

Means and variances use stats_dtype when provided. Supported values are
torch.float32, torch.float64, and None; the default chooses float64 on CPU and
float32 on CUDA before converting stored statistics to the input dtype.

After fitting, the estimator exposes:

- components_
- singular_values_
- mean_ and var_
- explained_variance_ and explained_variance_ratio_
- noise_variance_
- n_components_
- n_features_ and n_samples_seen_
- batch_size_ after fit()

The cached mean_proj_ is also available for the optimized transform path.

## Approximation semantics

Incremental PCA summarizes earlier batches by their retained components, so
results can depend on batch size and order even with the exact backend. They
should be numerically close to scikit-learn when batching, dtype, and component
count match.

The Gram backend targets the same singular triplets as full SVD, but forming a
Gram matrix squares the condition number. Its scale-aware loading and full-SVD
fallback protect degenerate cases without reporting ridge-shifted singular
values. Gram tail energy is computed from the original Frobenius norm.

The low-rank backend is intentionally approximate. Its explained variances use
the retained approximate singular values, while noise_variance_ uses total
Frobenius energy minus retained approximate energy over the full discarded
dimension. Consequently, residual noise remains meaningful even when
lowrank_q == n_components_.

## Benchmarking

The CUDA timing and optional scikit-learn comparison are executable code rather
than pytest tests:

~~~bash
python benchmarks/benchmark_backends.py
~~~

The regular test suite is run with:

~~~bash
python -m pytest -q
~~~

## Acknowledgments

The early implementation adapted code from David Ng's
[PCAonGPU](https://github.com/dnhkng/PCAonGPU), alongside scikit-learn's
IncrementalPCA implementation. This project has since added its own improvements
and extensions. The MIT license and copyright notices for PCAonGPU and this
project are preserved in [LICENSE](LICENSE).
