# IncrementalPCA improvement measurements

Tested in the updated `torch-incremental-pca` conda environment: Python 3.13.15,
PyTorch 2.14.0+cu130, CUDA 13.0, Triton 3.8.0, and an NVIDIA H100 80GB HBM3.
The original implementation is commit
`d962f0439818b0d59286ad218a5460bcdbd48e7d`. The preserved baseline was checked
byte-for-byte against that commit. Every raw result records SHA-256 hashes of
the baseline, updated estimator, benchmark script, and experimental kernels.

The changes primarily improve memory use, numerical behavior, and the API.
They do not produce a general fitting speedup in these measurements. Triton and
compiled-statistics variants remain optional benchmark experiments.

## Changes and validation

All four proposed API/statistics improvements are implemented:

- `fit` slices before conversion and transfer, with optional `compute_device`.
- Projection supports batching, output devices, writable tensor/NumPy/memmap
  buffers, and iterators that bound output memory.
- `inverse_transform`, defined handling of degenerate whitening directions, and
  `fit_transform` using a second pass through the final fitted basis.
- The delta-based Chan variance merge, reusing the mean difference in the
  augmented matrix and preserving `stats_dtype` across updates.

Other fixes cover rank-deficient Gram decompositions, float64 CUDA statistics,
large-offset projection cancellation, autocast, precision-setting restoration,
inferred Gram rank, retained tensor storage, and outputs consumed by trainable
layers. See [numerical regression tests](../../tests/test_numerical_improvements.py)
and [streaming/API tests](../../tests/test_streaming.py).

The final suite passed **111 tests, with no skips, in 5.42 seconds**, including
CPU and CUDA cases. Repository pre-commit checks passed. Two existing tests were
updated for the intentional change that CPU `mean_` and `var_` retain the default
float64 statistics dtype even when components are float32.

The benchmark separately validates all Triton prototypes at three shapes,
including non-tile-aligned dimensions. Centered projection is compared against
a float64 reference with offsets 0 and 1e6, whitening, and a zero scale.
Augmentation and merged statistics are checked against the PyTorch expressions.
These experiments support contiguous CUDA float32 inputs only.

## Method

Both main runs use five rounds and two untimed warm-up updates per round.
The larger shape times 12 further updates; the smaller shape times 24.
Reported update times are medians of synchronized wall-clock measurements in
milliseconds per `partial_fit`, including Python and prototype dispatch overhead.
Synthetic data, seed 123, and backend settings are shared by every variant.
PyTorch float32 matmul precision is `highest`, with two CPU threads.
Compilation and data generation are excluded.

Input is already on the GPU for fitting and projection timings. These are
measurements of two synthetic workloads on one GPU; they do not measure disk
I/O, CPU-to-GPU transfer throughput, or performance across devices and spectra.

The earlier three-round [large run](h100_torch214_initial.json) is also preserved.
Its original Gram timings ranged from 7.25 to 8.15 ms, motivating the five-round
repeat. The tables below use the complete five-round runs, with every round
available in [large raw results](h100_torch214.json) and
[small raw results](h100_torch214_small.json).

## Update latency

Batch 500, features 2000, retained components 100:

| Backend | Original ms | Updated ms | Triton augmentation ms | Triton preprocessing ms | Compiled merge ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 26.719 | 26.867 | 26.911 | 26.939 | 27.003 |
| gram | 7.203 | 7.373 | 7.393 | 7.353 | 7.422 |
| lowrank | 14.669 | 14.708 | 14.773 | 14.715 | 14.812 |

Batch 64, features 256, retained components 16:

| Backend | Original ms | Updated ms | Triton augmentation ms | Triton preprocessing ms | Compiled merge ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 2.648 | 2.694 | 2.748 | 2.684 | 2.858 |
| gram | 1.633 | 1.807 | 1.852 | 1.793 | 1.931 |
| lowrank | 3.299 | 3.374 | 3.405 | 3.353 | 3.484 |

The updated estimator costs about 0.3-2.4% at the larger shape. At the smaller
shape, full/randomized SVD cost about 1.7-2.3% more and Gram costs about 10.6%
more. The stronger numerical checks have a visible fixed cost on small updates.
Gram loading now avoids a scalar CPU-to-GPU transfer, but fixed overhead
remains.

The fused Triton preprocessing variant is less than 1% faster than updated
PyTorch in its best cases here. The compiled Chan merge does not improve total
update latency. These results do not justify automatic production dispatch.

## Projection, fusion, and memory

Public `transform` latency in microseconds per call:

| Shape (batch, features, components) | Original | Updated |
| --- | ---: | ---: |
| (500, 2000, 100) | 23.80 | 40.99 |
| (64, 256, 16) | 21.42 | 37.94 |

The added cost includes centering before matrix multiplication and API/context
handling. This avoids subtracting two large projected values for offset data.

Isolated microbenchmarks report median amortized wall time over five groups of
200 calls after ten warm-up calls. Values are microseconds, including launch
overhead. The centered Triton matmul uses explicit `tf32x3` arithmetic.

| Operation | Large PyTorch | Large Triton | Small PyTorch | Small Triton |
| --- | ---: | ---: | ---: | ---: |
| Augmented matrix assembly | 24.48 | 13.88 | 24.80 | 13.84 |
| Centered matrix multiplication | 22.68 | 51.40 | 17.58 | 18.04 |

Assembly is about 1.8 times faster in isolation. The decomposition and other
update work prevent this saving from improving total fit time materially.
The centered Triton matmul fails to beat PyTorch at either shape. Public
projection remains PyTorch in all fitting variants.

For a noncontiguous CUDA float16 source of shape `(4096, 512)`, with Gram rank 16
and fit batch size 128, peak additional PyTorch-allocated GPU memory falls from
**12.00 MiB to 1.65 MiB**. This measurement excludes the already-resident source
and measures allocated tensors, not total process VRAM or reserved allocator
memory. It includes fitting workspace as well as conversion storage.

At the larger fitting shape, retained component storage for full SVD falls from
**4,808,000 to 800,000 bytes**; randomized SVD falls from 1,600,000 to 800,000
bytes. Gram already owned compact component storage. Peak decomposition memory
does not necessarily decrease: compacting retained views requires a temporary
copy while the decomposition output is still live.

## Accuracy limits

The updated Gram basis agrees closely with the original span on these synthetic
inputs (minimum principal cosine above 0.9999997). Principal angles are computed
after float64 QR; these comparisons measure agreement with the original
incremental estimator, not exact PCA of all historical data.

Gram remains approximate and sensitive to conditioning. On the larger spectrum,
maximum absolute `components_ @ components_.T - I` is about 0.00104 for both
original and updated Gram, and about 0.00065 for the updated default CUDA full
SVD. The rank-deficiency fallback repairs the tested invalid null-space bases;
it does not guarantee direct-SVD accuracy for every spectrum. Raw results include
orthogonality, principal-angle, and singular-spectrum diagnostics for all variants.

## Reproduce

Activate the updated environment and run from the repository root:

```bash
conda activate torch-incremental-pca
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
git show d962f04:torch_incremental_pca/incremental_pca.py > /tmp/ipca_baseline.py
python -m pytest -q
python -m benchmarks.benchmark_improvements \
    --baseline /tmp/ipca_baseline.py --triton --compile-merge \
    --features 2000 --components 100 --batch-size 500 \
    --updates 12 --rounds 5 --output /tmp/ipca_large.json
python -m benchmarks.benchmark_improvements \
    --baseline /tmp/ipca_baseline.py --triton --compile-merge \
    --features 256 --components 16 --batch-size 64 \
    --updates 24 --rounds 5 --output /tmp/ipca_small.json
```
