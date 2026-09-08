"""Optional CUDA float32 Triton prototypes, kept outside the estimator API.

These kernels assume contiguous row-major inputs. Their validation and end-to-end
comparisons live in benchmark_improvements.py. Importing the package itself does
not import Triton. The centered matmul explicitly uses tf32x3 arithmetic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _augment(
    X,
    Mean,
    Delta,
    C,
    S,
    Out,
    factor,
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // D, offset % D
    live = offset < (K + B + 1) * D
    old = tl.load(C + row * D + col, live & (row < K), 0)
    scale = tl.load(S + row, live & (row < K), 0)
    batch = tl.load(X + (row - K) * D + col, live & (row >= K) & (row < K + B), 0)
    mean = tl.load(Mean + col, live, 0)
    delta = tl.load(Delta + col, live & (row == K + B), 0)
    value = tl.where(
        row < K, old * scale, tl.where(row < K + B, batch - mean, delta * -factor)
    )
    tl.store(Out + offset, value, live)


def assemble_augmented(X, mean, delta, components, singular_values, factor, out):
    B, D = X.shape
    K = components.shape[0]
    _augment[(triton.cdiv((K + B + 1) * D, 1024),)](
        X,
        mean,
        delta,
        components,
        singular_values,
        out,
        factor,
        B,
        D,
        K,
        1024,
    )
    return out


@triton.jit
def _centered_mm(
    X,
    Mean,
    C,
    Scale,
    Out,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    WHITEN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(D, BK)):
        features = block * BK + inner
        x = tl.load(
            X + rows[:, None] * D + features[None, :],
            (rows[:, None] < M) & (features[None, :] < D),
            0,
        )
        mean = tl.load(Mean + features, features < D, 0)
        c = tl.load(
            C + cols[None, :] * D + features[:, None],
            (cols[None, :] < N) & (features[:, None] < D),
            0,
        )
        acc = tl.dot(x - mean[None, :], c, acc, input_precision="tf32x3")
    if WHITEN:
        scale = tl.load(Scale + cols, cols < N, 0)
        acc *= scale[None, :]
    tl.store(
        Out + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def centered_transform(X, mean, components, inverse_scale=None):
    M, D = X.shape
    N = components.shape[0]
    out = torch.empty((M, N), device=X.device, dtype=X.dtype)
    _centered_mm[(triton.cdiv(M, 16), triton.cdiv(N, 32))](
        X,
        mean,
        components,
        inverse_scale if inverse_scale is not None else mean,
        out,
        M,
        N,
        D,
        inverse_scale is not None,
        16,
        32,
        64,
        num_warps=4,
    )
    return out


@triton.jit(do_not_specialize=["n1", "n2"])
def _prepare_update(
    X,
    BatchMean,
    BatchVar,
    OldMean,
    OldVar,
    C,
    S,
    Out,
    NewMean,
    NewVar,
    n1,
    n2,
    factor,
    B: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // D, offset % D
    live = offset < (K + B + 1) * D
    bmean = tl.load(BatchMean + col, live, 0)
    oldmean = tl.load(OldMean + col, live & ((row == 0) | (row == K + B)), 0)
    delta = bmean - oldmean
    old = tl.load(C + row * D + col, live & (row < K), 0)
    scale = tl.load(S + row, live & (row < K), 0)
    batch = tl.load(X + (row - K) * D + col, live & (row >= K) & (row < K + B), 0)
    value = tl.where(
        row < K, old * scale, tl.where(row < K + B, batch - bmean, delta * -factor)
    )
    tl.store(Out + offset, value, live)
    # Each feature's statistics are written once, by its first-row lane.
    first = live & (row == 0)
    oldvar = tl.load(OldVar + col, first, 0)
    bvar = tl.load(BatchVar + col, first, 0)
    n = n1 + n2
    mean = oldmean + (n2 / n) * delta
    var = (n1 * oldvar + n2 * bvar + (n1 * (n2 / n)) * delta * delta) / n
    tl.store(NewMean + col, mean, first)
    tl.store(NewVar + col, var, first)


def prepare_update(
    X,
    batch_mean,
    batch_var,
    old_mean,
    old_var,
    components,
    singular_values,
    n1,
    factor,
    out,
):
    B, D = X.shape
    K = components.shape[0]
    mean, var = torch.empty_like(old_mean), torch.empty_like(old_var)
    _prepare_update[(triton.cdiv((K + B + 1) * D, 1024),)](
        X,
        batch_mean,
        batch_var,
        old_mean,
        old_var,
        components,
        singular_values,
        out,
        mean,
        var,
        n1,
        B,
        factor,
        B,
        D,
        K,
        1024,
        enable_fp_fusion=False,
    )
    return out, mean, var, n1 + B
