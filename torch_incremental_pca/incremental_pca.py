from __future__ import annotations

import contextlib
import math
from typing import Optional, Tuple

import torch


class IncrementalPCA:
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages
    PyTorch for GPU acceleration.
    Adapted from
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py

    This class provides methods to fit the model on data incrementally in batches,
    and to transform new data based on the principal components learned during the
    fitting process.

    Three SVD backends are available, selectable via the `lowrank` and `gram` flags:

    - **Full SVD** (default): Uses `torch.linalg.svd`. Exact but uses Householder
      bidiagonalization which is inherently sequential and underutilizes GPU
      parallelism.
    - **Low-rank SVD** (`lowrank=True`): Uses `torch.svd_lowrank` (randomized SVD).
      Faster for very large matrices where `n_components << min(n_samples, n_features)`,
      but involves multiple power iterations that can be slower for moderate-sized
      matrices.
    - **Gram eigendecomposition** (`gram=True`): Computes `G = X @ X.T` (a GEMM) then
      `torch.linalg.eigh(G)` to recover singular values/vectors. Mathematically
      equivalent to full SVD but significantly faster on GPU because it replaces
      sequential Householder reflections with highly-parallelizable matrix
      multiplications. Recommended when the augmented matrix is wide (rows < cols),
      which is the typical case in incremental PCA when
      `n_components + batch_size < n_features`.

    Args:
        n_components (int, optional): Number of components to keep. If `None`, it's
            set to the minimum of the number of samples and features. Defaults to None.
        copy (bool): If False, this class may overwrite (mutate) input data in-place
            for performance (currently only on the first `partial_fit`, during
            centering). Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch.
            Only needed if self.fit is called. If `None`, it's inferred from the data
            and set to `5 * n_features`. Defaults to None.
        svd_driver (str, optional): name of the cuSOLVER method to be used for
            torch.linalg.svd. This keyword argument only works on CUDA inputs. Available
            options are: None, gesvd, gesvdj, and gesvda. Defaults to None.
        lowrank (bool, optional): Whether to use torch.svd_lowrank instead of
            torch.linalg.svd which can be faster. Mutually exclusive with `gram`.
            Defaults to False.
        lowrank_q (int, optional): For an adequate approximation of n_components,
            this parameter defaults to n_components * 2.
        lowrank_niter (int, optional): Number of subspace iterations to conduct for
            torch.svd_lowrank. Defaults to 4.
        lowrank_seed (int, optional): Seed for making results of torch.svd_lowrank
            reproducible.
        gram (bool, optional): Whether to use gram-matrix eigendecomposition instead of
            torch.linalg.svd. For wide matrices (rows < cols), this computes G = X @ X.T
            followed by torch.linalg.eigh(G) and recovers singular vectors via
            Vt = U.T @ X / S. Mathematically equivalent to full SVD but significantly
            faster on GPU because it uses GEMM operations instead of sequential
            Householder reflections. Falls back to full SVD when the matrix is tall
            (rows > cols). Mutually exclusive with `lowrank`. Defaults to False.
        stats_dtype (torch.dtype, optional): Data type to use for computing statistics
            (mean and variance). Defaults to None.
        ensure_contiguous (bool): Whether to enforce contiguous memory layout for inputs
            Defaults to True.
        gram_eps (float): Small epsilon value to avoid division by zero when computing
            inverse of singular values in gram mode. Defaults to 1e-7.
        allow_tf32 (bool, optional): Whether to allow TensorFloat-32 (TF32) execution
            on Ampere+ GPUs for matrix multiplications. Defaults to None.
        matmul_precision (str, optional): Matmul precision to use ("highest", "high",
            or "medium"). Requires PyTorch >= 2.0. Defaults to None.
        deterministic_flip (bool): Whether to apply SVD sign flipping deterministically.
            Defaults to True.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: bool = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        lowrank_seed: Optional[int] = None,
        gram: bool = False,
        # New knobs
        stats_dtype: Optional[torch.dtype] = None,
        ensure_contiguous: bool = True,
        gram_eps: float = 1e-7,
        # Perf knobs
        allow_tf32: Optional[bool] = None,
        matmul_precision: Optional[
            str
        ] = None,  # "highest" | "high" | "medium" (torch>=2.0)
        deterministic_flip: bool = True,
    ):
        self.n_components = n_components
        self.copy = copy
        self.batch_size = batch_size
        self.svd_driver = svd_driver

        self.lowrank = lowrank
        self.lowrank_q = lowrank_q
        self.lowrank_niter = lowrank_niter
        self.lowrank_seed = lowrank_seed

        self.gram = gram
        self.stats_dtype = stats_dtype
        self.ensure_contiguous = ensure_contiguous
        self.gram_eps = gram_eps

        self.allow_tf32 = allow_tf32
        self.matmul_precision = matmul_precision
        self.deterministic_flip = deterministic_flip

        self.n_features_ = None

        # Workspace for augmented matrix to reduce allocations
        self._x_aug_work: Optional[torch.Tensor] = None

        if self.lowrank and self.gram:
            raise ValueError(
                "lowrank and gram are mutually exclusive. Set only one to True."
            )
        if self.lowrank:
            self._validate_lowrank_params()

    def _validate_lowrank_params(self):
        if self.lowrank_q is None:
            if self.n_components is None:
                raise ValueError(
                    "n_components must be specified when using lowrank mode "
                    "with lowrank_q=None."
                )
            self.lowrank_q = self.n_components * 2
        elif self.n_components is not None and self.lowrank_q < self.n_components:
            raise ValueError("lowrank_q must be >= n_components.")

    @contextlib.contextmanager
    def _matmul_context(self):
        # Scoped TF32 / matmul precision toggles; restored afterwards.
        old_tf32 = None
        old_cudnn_tf32 = None
        old_prec = None
        changed_tf32 = self.allow_tf32 is not None and torch.cuda.is_available()
        changed_prec = self.matmul_precision is not None and hasattr(
            torch, "set_float32_matmul_precision"
        )

        try:
            if changed_tf32:
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = bool(self.allow_tf32)
                # cudnn TF32 can matter for some ops; harmless to mirror
                if hasattr(torch.backends, "cudnn") and hasattr(
                    torch.backends.cudnn, "allow_tf32"
                ):
                    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
                    torch.backends.cudnn.allow_tf32 = bool(self.allow_tf32)

            if changed_prec:
                # torch.get_float32_matmul_precision exists on modern PyTorch
                if hasattr(torch, "get_float32_matmul_precision"):
                    old_prec = torch.get_float32_matmul_precision()
                torch.set_float32_matmul_precision(self.matmul_precision)

            yield
        finally:
            if (
                changed_prec
                and old_prec is not None
                and hasattr(torch, "set_float32_matmul_precision")
            ):
                torch.set_float32_matmul_precision(old_prec)
            if changed_tf32 and old_tf32 is not None:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32
            if (
                changed_tf32
                and old_cudnn_tf32 is not None
                and hasattr(torch.backends, "cudnn")
            ):
                torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    def _svd_fn_full(self, X):
        return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)

    def _svd_fn_lowrank(self, X):
        seed_enabled = self.lowrank_seed is not None
        with torch.random.fork_rng(enabled=seed_enabled):
            if seed_enabled:
                torch.manual_seed(self.lowrank_seed)
            U, S, V = torch.svd_lowrank(X, q=self.lowrank_q, niter=self.lowrank_niter)
            return U, S, V.mH

    def _svd_fn_gram_topk(self, X):
        """
        Wide-matrix fast path: G = X @ X.T then eigh(G), recover Vt.
        Avoids flipping full eigensystem; slices only top-k.
        Also fuses invS scaling into the small (k x m) factor before GEMM.
        """
        m, D = X.shape
        if m > D:
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        k = min(self.n_components or m, m)

        # G is (m, m)
        G = X @ X.mT
        G += self.gram_eps * torch.eye(m, device=G.device, dtype=G.dtype)
        evals, evecs = torch.linalg.eigh(G)  # ascending

        # Take largest-k (from the end) then flip just those to descending
        evals_k = evals[-k:].flip(0)  # (k,)
        U_k = evecs[:, -k:].flip(1)  # (m, k)

        S_k = torch.sqrt(evals_k.clamp(min=0))

        invS = S_k.reciprocal()  # (k,)

        # Fuse scaling into small factor (k x m) then GEMM to get (k x D)
        # Vt_k = diag(1/S) @ U^T @ X
        Vt_k = (invS[:, None] * U_k.mT) @ X

        tail_count = m - k
        if tail_count > 0:
            # tail are the smallest m-k eigenvalues (ascending => evals[:-k])
            tail_ss = evals[:-k].clamp(min=0).sum()
        else:
            tail_ss = torch.zeros((), device=X.device, dtype=X.dtype)

        return U_k, S_k, Vt_k, tail_ss, tail_count

    def _validate_fit_batch(self, X) -> torch.Tensor:
        valid_dtypes = (torch.float32, torch.float64)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        # NOTE: We no longer clone for copy=True. Inputs are only modified when
        # copy=False (first-pass centering).

        if self.ensure_contiguous and not X.is_contiguous():
            X = X.contiguous()

        n_samples, n_features = X.shape
        if self.n_components is not None:
            if self.n_components > n_features:
                raise ValueError(
                    f"n_components={self.n_components} invalid "
                    f"for n_features={n_features}."
                )
            if self.n_components > n_samples:
                raise ValueError(
                    f"n_components={self.n_components} must be <= "
                    f"batch n_samples={n_samples}."
                )

        if X.dtype not in valid_dtypes:
            X = X.to(torch.float32)

        return X

    def _validate_transform(self, X) -> torch.Tensor:
        if not hasattr(self, "components_"):
            raise ValueError("IncrementalPCA instance is not fitted yet.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(
                X, dtype=self.components_.dtype, device=self.components_.device
            )
        else:
            if X.device != self.components_.device:
                raise ValueError(
                    f"X is on device {X.device}, "
                    f"but model is on {self.components_.device}."
                )
            if X.dtype != self.components_.dtype:
                X = X.to(self.components_.dtype)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted "
                f"with {self.n_features_}."
            )

        if self.ensure_contiguous and not X.is_contiguous():
            X = X.contiguous()

        return X

    @staticmethod
    def _incremental_mean_and_var(
        X: torch.Tensor,
        last_mean: Optional[torch.Tensor],
        last_variance: Optional[torch.Tensor],
        last_sample_count: int,
        *,
        stats_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean (D,), var (D,), total_count (int), batch_mean (D,), batch_var (D,)
        """
        n2 = int(X.shape[0])
        if n2 == 0:
            if last_mean is None or last_variance is None:
                raise ValueError("Empty batch with uninitialized statistics.")
            # batch_mean/var are undefined; return last as batch too
            return last_mean, last_variance, last_sample_count, last_mean, last_variance

        input_dtype = X.dtype
        Xs = X if X.dtype == stats_dtype else X.to(stats_dtype)

        batch_var, batch_mean = torch.var_mean(Xs, dim=0, unbiased=False)

        if last_sample_count == 0 or last_mean is None or last_variance is None:
            mean = batch_mean
            var = batch_var
            n = n2
        else:
            n1 = last_sample_count
            m1 = last_mean.to(stats_dtype)
            v1 = last_variance.to(stats_dtype)
            m2 = batch_mean
            v2 = batch_var
            n = n1 + n2

            mean = (m1 * n1 + m2 * n2) / n
            d1 = m1 - mean
            d2 = m2 - mean
            ss = n1 * (v1 + d1.square()) + n2 * (v2 + d2.square())
            var = ss / n

        mean_out = mean.to(input_dtype)
        var_out = var.to(input_dtype)
        batch_mean_out = batch_mean.to(input_dtype)
        batch_var_out = batch_var.to(input_dtype)
        return mean_out, var_out, int(n), batch_mean_out, batch_var_out

    @staticmethod
    def _svd_flip(u: torch.Tensor, v: torch.Tensor, u_based_decision: bool = True):
        # In-place sign correction on SVD outputs (not on inputs).
        if u_based_decision:
            max_abs_rows = torch.argmax(u.abs(), dim=0)
            cols = torch.arange(u.shape[1], device=u.device)
            signs = torch.sign(u[max_abs_rows, cols])
        else:
            max_abs_cols = torch.argmax(v.abs(), dim=1)
            rows = torch.arange(v.shape[0], device=v.device)
            signs = torch.sign(v[rows, max_abs_cols])

        signs = torch.where(signs == 0, torch.ones_like(signs), signs)

        u *= signs[: u.shape[1]].view(1, -1)
        v *= signs.view(-1, 1)
        return u, v

    def _get_x_aug_work(self, m: int, n_features: int, device, dtype) -> torch.Tensor:
        need_new = (
            self._x_aug_work is None
            or self._x_aug_work.device != device
            or self._x_aug_work.dtype != dtype
            or self._x_aug_work.shape[1] != n_features
            or self._x_aug_work.shape[0] < m
        )
        if need_new:
            self._x_aug_work = torch.empty((m, n_features), device=device, dtype=dtype)
        return self._x_aug_work[:m]

    @torch.inference_mode()
    def fit(self, X, check_input: bool = True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to
                True.

        Returns:
            IncrementalPCA: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_fit_batch(X)
        n_samples, n_features = X.shape

        if self.batch_size is None:
            if self.gram:
                k = self.n_components or 0
                max_batch_for_wide = max(1, n_features - k - 1)
                self.batch_size = min(5 * n_features, max_batch_for_wide)
                if self.n_components is not None:
                    # Ensure the first batch can learn the requested number of
                    # components. If this violates the wide-matrix condition,
                    # gram SVD will fall back to full SVD internally.
                    self.batch_size = max(self.batch_size, self.n_components)
            else:
                self.batch_size = 5 * n_features

        if self.n_components is not None and self.batch_size < self.n_components:
            raise ValueError(
                f"batch_size={self.batch_size} must be "
                f">= n_components={self.n_components}."
            )

        for batch in self.gen_batches(
            n_samples, self.batch_size, min_batch_size=self.n_components or 0
        ):
            self.partial_fit(X[batch], check_input=False)
        return self

    @torch.inference_mode()
    def partial_fit(self, X, check_input: bool = True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples,
                n_features).
            check_input (bool, optional): If True, validates the input. Defaults to
                True.

        Returns:
            IncrementalPCA: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_fit_batch(X)
        else:
            if self.ensure_contiguous and not X.is_contiguous():
                X = X.contiguous()
            if X.dtype not in (torch.float32, torch.float64):
                X = X.to(torch.float32)

        n_samples, n_features = X.shape

        if first_pass:
            self.mean_ = None
            self.var_ = None
            self.n_samples_seen_ = 0  # python int
            self.n_features_ = n_features
            if not self.n_components:
                self.n_components = min(n_samples, n_features)

        if n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match the first batch."
            )

        stats_dtype = (
            self.stats_dtype
            if self.stats_dtype is not None
            else (torch.float32 if X.is_cuda else torch.float64)
        )

        col_mean, col_var, n_total_samples, batch_mean, _batch_var = (
            self._incremental_mean_and_var(
                X, self.mean_, self.var_, self.n_samples_seen_, stats_dtype=stats_dtype
            )
        )

        # Build the matrix to decompose:
        if first_pass:
            if self.copy:
                # Center (out-of-place) to avoid modifying input; this is one pass.
                X_for_svd = X - col_mean
            else:
                # In-place centering for performance when caller allows mutation.
                X.sub_(col_mean)
                X_for_svd = X
        else:
            # Mean correction term
            factor = math.sqrt((self.n_samples_seen_ / n_total_samples) * n_samples)
            mean_correction = (self.mean_ - batch_mean) * factor  # (D,)

            k = self.n_components
            m = k + n_samples + 1

            X_aug = self._get_x_aug_work(m, n_features, device=X.device, dtype=X.dtype)

            # Top block: components_ * singular_values_ (fused into output)
            torch.mul(self.components_, self.singular_values_[:, None], out=X_aug[:k])

            # Middle block: centered batch written directly (no temp)
            torch.sub(X, batch_mean, out=X_aug[k : k + n_samples])

            # Last row: mean correction
            X_aug[-1].copy_(mean_correction)

            X_for_svd = X_aug

        # Decomposition (optionally with TF32)
        tail_ss = tail_count = None
        with self._matmul_context():
            if self.lowrank:
                U, S, Vt = self._svd_fn_lowrank(X_for_svd)
            elif self.gram:
                U, S, Vt, tail_ss, tail_count = self._svd_fn_gram_topk(X_for_svd)
            else:
                U, S, Vt = self._svd_fn_full(X_for_svd)

        if self.deterministic_flip:
            # Prefer u-based decision to avoid expensive argmax over D.
            U, Vt = self._svd_flip(U, Vt, u_based_decision=True)

        S2 = S.square()
        denom = col_var.sum() * n_total_samples

        if n_total_samples > 1:
            explained_variance = S2 / (n_total_samples - 1)
        else:
            explained_variance = torch.zeros_like(S2)

        explained_variance_ratio = S2 / denom
        explained_variance_ratio = torch.where(
            torch.isfinite(explained_variance_ratio),
            explained_variance_ratio,
            torch.zeros_like(explained_variance_ratio),
        )

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]

        # Precompute mean projection for transform (avoids allocating X-mean)
        # shape: (k,)
        self.mean_proj_ = self.mean_ @ self.components_.T

        # noise variance
        if tail_ss is not None and tail_count is not None:
            if tail_count > 0 and n_total_samples > 1:
                self.noise_variance_ = (tail_ss / (n_total_samples - 1)) / tail_count
            else:
                self.noise_variance_ = torch.zeros((), device=X.device, dtype=X.dtype)
        else:
            if S.numel() > self.n_components:
                self.noise_variance_ = explained_variance[self.n_components :].mean()
            else:
                self.noise_variance_ = torch.zeros((), device=X.device, dtype=X.dtype)

        return self

    @torch.inference_mode()
    def transform(self, X) -> torch.Tensor:
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously
        extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to
                be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X = self._validate_transform(X)
        with self._matmul_context():
            # Avoid allocating (X - mean) by using precomputed mean projection
            return (X @ self.components_.T) - self.mean_proj_

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generator to create slices containing `batch_size` elements from 0 to `n`.

        The last slice may contain less than `batch_size` elements,
        when `batch_size` does not divide `n`.

        Args:
            n (int): Size of the sequence.
            batch_size (int): Number of elements in each batch.
            min_batch_size (int, optional): Minimum number of elements in each batch.
                Defaults to 0.

        Yields:
            slice: A slice of `batch_size` elements.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        start = 0
        while start < n:
            end = min(start + batch_size, n)
            if n - end < min_batch_size:
                end = n
            yield slice(start, end)
            start = end
