from __future__ import annotations

import contextlib
import math
from numbers import Integral, Real
from typing import Iterator, Optional, Tuple

import torch


class IncrementalPCA:
    """Incremental principal component analysis for CPU and CUDA tensors.

    The update follows scikit-learn's augmented-matrix incremental SVD:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py

    Args:
        n_components: Retained rank. None infers the rank from the first batch.
            Automatic fit batching allows min(n_samples, n_features) components.
        copy: Preserve input batches when True. False allows centering a writable
            first batch in place. Read-only arrays are copied one batch at a time.
        batch_size: Fit batch size. The default is 5 * n_features for full and
            randomized SVD, or a wide-matrix size in Gram mode when possible.
        svd_driver: Optional CUDA torch.linalg.svd driver. PyTorch's default uses
            gesvdj with a gesvd fallback. gesvda is approximate.
        lowrank: Use randomized torch.svd_lowrank, exclusive with gram.
        lowrank_q: Randomized subspace size, defaulting to twice the retained rank
            and capped by matrix dimensions.
        lowrank_niter: Randomized power iteration count, default 4.
        lowrank_seed: Optional seed with scoped RNG state restoration.
        gram: Use a wide matrix's X @ X.T eigensystem. This squares the condition
            number, so numerically unreliable retained directions fall back to
            full SVD. Tall matrices also use full SVD.
        stats_dtype: Persistent mean/variance precision. Defaults to float64 on
            CPU and the computation dtype on CUDA; float64 input is not demoted.
        ensure_contiguous: Make each input batch contiguous before computation.
        gram_eps: Positive absolute singular-value floor, combined with a
            scale-aware Gram eigenvalue reliability check. Default 1e-7.
        allow_tf32: Optional scoped float32 matmul setting. Cannot conflict with
            matmul_precision. These settings are process-wide, not thread-local.
        matmul_precision: Optional highest, high, or medium float32 matmul mode.
        deterministic_flip: Apply a consistent sign convention to retained axes.
        compute_device: Optional device for per-batch input transfers. None uses
            the first batch's device. Refitting can establish a new device/dtype.
        whiten: Scale scores using learned variances. Degenerate coordinates are
            zeroed; inverse_transform applies the corresponding inverse scaling.
        whiten_eps: Relative cutoff on component standard deviation, compared
            with the largest retained standard deviation. Default 1e-7.

    Fitting and projection disable autograd and autocast. Returned projections
    are normal tensors and can feed a trainable downstream model. Components
    keep the first batch's float32/float64 compute dtype, while statistics retain
    stats_dtype. fit and transform accept sliceable NumPy/memmap/lazy sources;
    transform_batches bounds result storage as well as input storage.
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
        *,
        compute_device: Optional[torch.device | str] = None,
        whiten: bool = False,
        whiten_eps: float = 1e-7,
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
        self.compute_device = (
            torch.device(compute_device) if compute_device is not None else None
        )
        self.whiten = whiten
        self.whiten_eps = whiten_eps

        self._reset_fit_state()
        self._validate_parameters()

    def _reset_fit_state(self):
        """Remove all state learned from earlier calls to fit."""
        learned_attributes = (
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
            "_whitening_cache",
        )
        for attribute in learned_attributes:
            self.__dict__.pop(attribute, None)

        # Workspace for the augmented matrix; it must not survive a refit on a
        # different shape, device, or dtype.
        self._x_aug_work: Optional[torch.Tensor] = None

    @staticmethod
    def _is_positive_integer(value) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool) and value > 0

    def _validate_parameters(self):
        if self.n_components is not None and not self._is_positive_integer(
            self.n_components
        ):
            raise ValueError("n_components must be a positive integer or None.")
        if self.batch_size is not None and not self._is_positive_integer(
            self.batch_size
        ):
            raise ValueError("batch_size must be a positive integer or None.")
        if self.lowrank_q is not None and not self._is_positive_integer(self.lowrank_q):
            raise ValueError("lowrank_q must be a positive integer or None.")
        if (
            not isinstance(self.lowrank_niter, Integral)
            or isinstance(self.lowrank_niter, bool)
            or self.lowrank_niter < 0
        ):
            raise ValueError("lowrank_niter must be a nonnegative integer.")
        if (
            not isinstance(self.gram_eps, Real)
            or isinstance(self.gram_eps, bool)
            or not math.isfinite(float(self.gram_eps))
            or self.gram_eps <= 0
        ):
            raise ValueError("gram_eps must be finite and strictly positive.")
        if self.stats_dtype not in (None, torch.float32, torch.float64):
            raise ValueError(
                "stats_dtype must be torch.float32, torch.float64, or None."
            )
        if self.matmul_precision not in (None, "highest", "high", "medium"):
            raise ValueError("matmul_precision must be highest, high, medium, or None.")
        if (
            self.allow_tf32 is not None
            and self.matmul_precision is not None
            and bool(self.allow_tf32) != (self.matmul_precision != "highest")
        ):
            raise ValueError(
                "allow_tf32 and matmul_precision specify conflicting precision."
            )
        if (
            not isinstance(self.whiten_eps, Real)
            or isinstance(self.whiten_eps, bool)
            or not math.isfinite(float(self.whiten_eps))
            or self.whiten_eps <= 0
        ):
            raise ValueError("whiten_eps must be finite and strictly positive.")
        if self.lowrank and self.gram:
            raise ValueError(
                "lowrank and gram are mutually exclusive. Set only one to True."
            )
        if (
            self.lowrank_q is not None
            and self.n_components is not None
            and self.lowrank_q < self.n_components
        ):
            raise ValueError("lowrank_q must be >= n_components.")

    @contextlib.contextmanager
    def _matmul_context(self):
        # The legacy TF32 flag and matmul precision control the same state.
        # Save/restore the complete precision, including "medium". PCA does not
        # need to alter the unrelated cuDNN convolution policy.
        precision = self.matmul_precision
        if precision is None and self.allow_tf32 is not None:
            precision = "high" if self.allow_tf32 else "highest"
        if precision is None:
            yield
            return
        old_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision(precision)
            yield
        finally:
            torch.set_float32_matmul_precision(old_precision)

    def _svd_fn_full(self, X):
        return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)

    def _svd_fn_lowrank(self, X):
        q = self.lowrank_q
        if q is None:
            q = self.n_components_ * 2
        q = min(q, min(X.shape))
        if q < self.n_components_:
            raise ValueError("lowrank_q must be >= n_components_.")

        seed_enabled = self.lowrank_seed is not None
        with torch.random.fork_rng(enabled=seed_enabled):
            if seed_enabled:
                torch.manual_seed(self.lowrank_seed)
            U, S, V = torch.svd_lowrank(X, q=q, niter=self.lowrank_niter)
            return U, S, V.mH

    def _svd_fn_gram_topk(self, X):
        """Recover retained singular triplets from a wide matrix's Gram system."""
        m, D = X.shape
        if m > D:
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        rank = getattr(self, "n_components_", self.n_components)
        k = min(rank if rank is not None else m, m)
        G = X @ X.mT
        max_abs_diagonal = G.diagonal().abs().max()
        loading = (torch.finfo(G.dtype).eps * m * max_abs_diagonal).clamp_min(
            float(self.gram_eps) ** 2
        )
        G.diagonal().add_(loading)
        try:
            evals, evecs = torch.linalg.eigh(G)
        except torch.linalg.LinAlgError:
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        U_k = evecs[:, -k:].flip(1)
        Y = U_k.mT @ X
        S_k = torch.linalg.vector_norm(Y, dim=1)
        # An absolute singular-value check accepts roundoff as a null-space
        # direction on scaled/rank-deficient inputs. Test the unshifted retained
        # eigenvalue against the Gram rounding scale too. One host sync suffices.
        reliable = (
            torch.isfinite(S_k).all()
            & (S_k.min() > self.gram_eps)
            & (evals[-k] - loading > loading)
        )
        if not bool(reliable):
            U, S, Vt = self._svd_fn_full(X)
            return U, S, Vt, None, None

        # Norm recovery can slightly reorder almost-tied eigenvalues.
        S_k, order = S_k.sort(descending=True)
        U_k = U_k[:, order]
        Vt_k = Y[order] / S_k[:, None]
        tail_count = m - k
        tail_ss = (
            (torch.linalg.vector_norm(X).square() - S_k.square().sum()).clamp(min=0)
            if tail_count > 0
            else X.new_zeros(())
        )
        return U_k, S_k, Vt_k, tail_ss, tail_count

    @staticmethod
    def _source_shape(X, *, allow_empty=False):
        """Inspect an array/lazy source without converting or loading its contents."""
        shape = getattr(X, "shape", None)
        if shape is None:
            try:
                shape = (len(X), len(X[0]))
            except (TypeError, IndexError, KeyError) as exc:
                raise ValueError("X must be a nonempty 2D input.") from exc
        if len(shape) != 2:
            raise ValueError(f"X must be a 2D input; got {len(shape)} dimensions.")
        n_samples, n_features = shape
        if n_features == 0 or (n_samples == 0 and not allow_empty):
            raise ValueError("X must be nonempty in both dimensions.")
        return int(n_samples), int(n_features)

    @staticmethod
    def _as_tensor_batch(X):
        if isinstance(X, torch.Tensor):
            return X
        # A read-only mmap cannot safely back an in-place PyTorch operation.
        # This copy is bounded by the already-sliced batch, not the dataset.
        flags = getattr(X, "flags", None)
        strides = getattr(X, "strides", ())
        if (flags is not None and not flags.writeable) or any(s < 0 for s in strides):
            X = X.copy()
        return torch.as_tensor(X)

    def _validate_fit_batch(self, X) -> torch.Tensor:
        tensor_input = isinstance(X, torch.Tensor)
        X = self._as_tensor_batch(X)
        self._source_shape(X)
        if X.is_complex():
            raise ValueError("X must contain real-valued data.")
        if self.n_components is not None and self.n_components > X.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} invalid for n_features={X.shape[1]}."
            )
        fitted = hasattr(self, "components_")
        target_device = self.compute_device
        if fitted:
            if (
                target_device is None
                and tensor_input
                and X.device != self.components_.device
            ):
                raise ValueError(
                    f"X is on device {X.device}, "
                    f"but model is on {self.components_.device}."
                )
            target_device = self.components_.device
            dtype = self.components_.dtype
        else:
            dtype = (
                X.dtype if X.dtype in (torch.float32, torch.float64) else torch.float32
            )
        X = X.to(device=target_device or X.device, dtype=dtype)
        if self.ensure_contiguous and not X.is_contiguous():
            X = X.contiguous()
        return X

    def _validate_transform(self, X, *, inverse=False, allow_transfer=False):
        if not hasattr(self, "components_"):
            raise ValueError("IncrementalPCA instance is not fitted yet.")
        if (
            isinstance(X, torch.Tensor)
            and X.device != self.components_.device
            and not allow_transfer
            and self.compute_device is None
        ):
            raise ValueError(
                f"X is on device {X.device}, but model is on {self.components_.device}."
            )
        X = self._as_tensor_batch(X)
        _, features = self._source_shape(X, allow_empty=True)
        expected = self.n_components_ if inverse else self.n_features_
        if features != expected:
            raise ValueError(f"X has {features} features, but expected {expected}.")
        if X.is_complex():
            raise ValueError("X must contain real-valued data.")
        X = X.to(device=self.components_.device, dtype=self.components_.dtype)
        if self.ensure_contiguous and not X.is_contiguous():
            X = X.contiguous()
        return X

    @classmethod
    def _incremental_mean_and_var(
        cls,
        X: torch.Tensor,
        last_mean: Optional[torch.Tensor],
        last_variance: Optional[torch.Tensor],
        last_sample_count: int,
        *,
        stats_dtype: torch.dtype,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Returns:
            mean, var, total_count, batch_mean, batch_var, batch_mean - last_mean
            (the last value is None on the first batch).
        """
        n2 = int(X.shape[0])
        if n2 == 0:
            if last_mean is None or last_variance is None:
                raise ValueError("Empty batch with uninitialized statistics.")
            # batch_mean/var are undefined; return last as batch too
            return (
                last_mean,
                last_variance,
                last_sample_count,
                last_mean,
                last_variance,
                None,
            )

        Xs = X if X.dtype == stats_dtype else X.to(stats_dtype)
        batch_var, batch_mean = torch.var_mean(Xs, dim=0, unbiased=False)
        if last_sample_count == 0 or last_mean is None or last_variance is None:
            return batch_mean, batch_var, n2, batch_mean, batch_var, None

        n1 = last_sample_count
        n = n1 + n2
        previous_mean = last_mean.to(stats_dtype)
        previous_var = last_variance.to(stats_dtype)
        mean, var, delta = cls._merge_statistics(
            previous_mean, previous_var, batch_mean, batch_var, n1, n2
        )
        return mean, var, n, batch_mean, batch_var, delta

    @staticmethod
    def _merge_statistics(previous_mean, previous_var, batch_mean, batch_var, n1, n2):
        """Pure tensor Chan merge for optional compilation experiments."""
        n = n1 + n2
        delta = batch_mean - previous_mean
        mean = previous_mean + (n2 / n) * delta
        m2 = n1 * previous_var + n2 * batch_var + (n1 * (n2 / n)) * delta.square()
        # Do not cast persistent statistics back to the decomposition dtype.
        return mean, m2 / n, delta

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

    def _build_augmented(self, X, batch_mean, mean_delta, factor):
        k = self.n_components_
        n_samples, n_features = X.shape
        augmented = self._get_x_aug_work(
            k + n_samples + 1, n_features, device=X.device, dtype=X.dtype
        )
        torch.mul(self.components_, self.singular_values_[:, None], out=augmented[:k])
        torch.sub(X, batch_mean, out=augmented[k : k + n_samples])
        torch.mul(mean_delta, -factor, out=augmented[-1])
        return augmented

    @torch.no_grad()
    def fit(self, X, check_input: bool = True):
        """Fit a sliceable 2D source, converting/transferring only one batch at a time.

        X may be a tensor, a NumPy array/memmap, or a lazy source with ``shape``
        and row slicing. ``compute_device`` controls the optional batch transfer.
        Earlier fitted state is reset. With ``copy=False``, a writable first
        batch may be centered in place.
        """
        self._reset_fit_state()
        self._validate_parameters()
        n_samples, n_features = self._source_shape(X)
        k = (
            self.n_components
            if self.n_components is not None
            else min(n_samples, n_features)
        )
        if k > n_features:
            raise ValueError(f"n_components={k} invalid for n_features={n_features}.")
        if self.batch_size is None:
            self.batch_size_ = (
                max(k, n_features - k - 1) if self.gram else 5 * n_features
            )
        else:
            self.batch_size_ = self.batch_size
        if self.n_components is not None and self.batch_size_ < self.n_components:
            raise ValueError(
                f"batch_size={self.batch_size_} must be "
                f">= n_components={self.n_components}."
            )
        for batch in self.gen_batches(
            n_samples, self.batch_size_, min_batch_size=self.n_components or 0
        ):
            self.partial_fit(X[batch], check_input=check_input)
        return self

    @torch.no_grad()
    def partial_fit(self, X, check_input: bool = True):
        """Update from one batch. The first batch must contain at least k rows.

        Basic shape/device/dtype checks are always performed, including when
        ``check_input=False``. Later batches use the first batch's compute dtype.
        Autocast is disabled for the numerical update.
        """
        self._validate_parameters()
        X = self._validate_fit_batch(X)
        with torch.autocast(device_type=X.device.type, enabled=False):
            return self._partial_fit(X)

    def _prepare_update(self, X, stats_dtype, first_pass):
        n_samples = X.shape[0]
        col_mean, col_var, n_total_samples, batch_mean, _batch_var, mean_delta = (
            self._incremental_mean_and_var(
                X, self.mean_, self.var_, self.n_samples_seen_, stats_dtype=stats_dtype
            )
        )

        # Build the matrix to decompose:
        if first_pass:
            if self.copy:
                # Center (out-of-place) to avoid modifying input; this is one pass.
                X_for_svd = torch.empty_like(X)
                torch.sub(X, col_mean, out=X_for_svd)
            else:
                # In-place centering for performance when caller allows mutation.
                X.sub_(col_mean)
                X_for_svd = X
        else:
            # Mean correction term
            factor = math.sqrt((self.n_samples_seen_ / n_total_samples) * n_samples)
            X_for_svd = self._build_augmented(X, batch_mean, mean_delta, factor)

        return X_for_svd, col_mean, col_var, n_total_samples

    def _partial_fit(self, X):
        first_pass = not hasattr(self, "components_")
        n_samples, n_features = X.shape

        if first_pass:
            self.mean_ = None
            self.var_ = None
            self.n_samples_seen_ = 0  # python int
            self.n_features_ = n_features
            self.n_components_ = (
                self.n_components
                if self.n_components is not None
                else min(n_samples, n_features)
            )
            if self.n_components_ > n_samples:
                raise ValueError(
                    f"n_components={self.n_components_} must be <= "
                    f"the first batch n_samples={n_samples}."
                )

        if n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match the first batch."
            )

        stats_dtype = (
            self.stats_dtype
            if self.stats_dtype is not None
            else (X.dtype if X.is_cuda else torch.float64)
        )

        X_for_svd, col_mean, col_var, n_total_samples = self._prepare_update(
            X, stats_dtype, first_pass
        )

        # Decomposition (optionally with TF32)
        tail_ss = tail_count = None
        with self._matmul_context():
            if self.lowrank:
                U, S, Vt = self._svd_fn_lowrank(X_for_svd)
            elif self.gram:
                U, S, Vt, tail_ss, tail_count = self._svd_fn_gram_topk(X_for_svd)
            else:
                U, S, Vt = self._svd_fn_full(X_for_svd)

        k = self.n_components_
        components = Vt[:k].clone(memory_format=torch.contiguous_format)
        if self.deterministic_flip:
            # Only retained vectors matter. U is read for the sign decision;
            # the discarded left singular vectors need no in-place update.
            U_k = U[:, :k]
            rows = U_k.abs().argmax(dim=0)
            signs = U_k[rows, torch.arange(k, device=X.device)].sign()
            components.mul_(torch.where(signs == 0, 1, signs)[:, None])

        singular_values = S[:k].clone()
        S2 = singular_values.square()
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
        self.components_ = components
        self.singular_values_ = singular_values
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]

        # Retain the projected mean for compatibility. Stable transform centers
        # before multiplication instead of using this cache.
        self.mean_proj_ = self.mean_.to(self.components_.dtype) @ self.components_.T
        self._whitening_cache = None

        # noise variance
        discarded_count = min(X_for_svd.shape) - self.n_components_
        if discarded_count > 0 and n_total_samples > 1:
            if tail_ss is not None and tail_count is not None:
                residual_ss = tail_ss
                discarded_count = tail_count
            elif self.lowrank:
                residual_ss = (
                    torch.linalg.vector_norm(X_for_svd).square() - S2.sum()
                ).clamp(min=0)
            else:
                residual_ss = S[self.n_components_ :].square().sum()
            self.noise_variance_ = residual_ss / (
                (n_total_samples - 1) * discarded_count
            )
        else:
            self.noise_variance_ = torch.zeros((), device=X.device, dtype=X.dtype)

        return self

    def _whitening_factors(self):
        if self._whitening_cache is None:
            std = (
                self.explained_variance_.to(self.components_.dtype).clamp(min=0).sqrt()
            )
            keep = std > self.whiten_eps * std.max()
            scale = torch.where(keep, std, 0)
            inverse = torch.where(keep, torch.where(keep, std, 1).reciprocal(), 0)
            self._whitening_cache = scale, inverse
        return self._whitening_cache

    @torch.no_grad()
    def _project_batch(self, X, *, inverse=False, allow_transfer=False):
        X = self._validate_transform(X, inverse=inverse, allow_transfer=allow_transfer)
        with (
            torch.autocast(device_type=X.device.type, enabled=False),
            self._matmul_context(),
        ):
            if inverse:
                scores = X * self._whitening_factors()[0] if self.whiten else X
                result = scores @ self.components_
                torch.add(result, self.mean_, out=result)
            else:
                # Subtract before the GEMM: subtracting a projected mean can
                # catastrophically cancel for data with a large offset.
                centered = torch.empty_like(X)
                torch.sub(X, self.mean_, out=centered)
                result = centered @ self.components_.mT
                if self.whiten:
                    result.mul_(self._whitening_factors()[1])
        return result

    def _transform_metadata(self, X, batch_size, inverse):
        if not hasattr(self, "components_"):
            raise ValueError("IncrementalPCA instance is not fitted yet.")
        n, features = self._source_shape(X, allow_empty=True)
        expected = self.n_components_ if inverse else self.n_features_
        if features != expected:
            raise ValueError(f"X has {features} features, but expected {expected}.")
        if batch_size is None:
            batch_size = getattr(self, "batch_size_", self.batch_size) or 1024
        if not self._is_positive_integer(batch_size):
            raise ValueError("batch_size must be a positive integer.")
        return n, batch_size

    def _output_device(self, output_device):
        device = (
            torch.device(output_device)
            if output_device is not None
            else self.components_.device
        )
        if device.type == "cpu":
            return torch.device("cpu")
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def _projection_batches(self, X, batch_size, output_device, *, inverse=False):
        n, batch_size = self._transform_metadata(X, batch_size, inverse)
        device = self._output_device(output_device)
        for batch in self.gen_batches(n, batch_size):
            # The no-grad/autocast contexts end inside _project_batch, before
            # yielding, so the caller's execution context is never changed.
            yield self._project_batch(
                X[batch], inverse=inverse, allow_transfer=True
            ).to(device)

    def transform_batches(
        self, X, *, batch_size=None, output_device=None
    ) -> Iterator[torch.Tensor]:
        """Yield independently owned score batches, optionally transferred to CPU.

        Both input conversion and output storage are bounded by a batch while
        the consumer releases completed batches. Tensor sources may be on CPU
        even when the model is on CUDA. Iteration uses the fitted basis; do not
        call partial_fit while consuming the iterator.
        """
        return self._projection_batches(X, batch_size, output_device)

    def inverse_transform_batches(
        self, X, *, batch_size=None, output_device=None
    ) -> Iterator[torch.Tensor]:
        """Yield reconstructed feature batches using the fitted basis."""
        return self._projection_batches(X, batch_size, output_device, inverse=True)

    @torch.no_grad()
    def _transform_into(self, X, batch_size, output_device, out, *, inverse=False):
        n, size = self._transform_metadata(X, batch_size, inverse)
        width = self.n_features_ if inverse else self.n_components_
        device = self._output_device(output_device)
        if out is None and 0 < n <= size:
            return self._project_batch(
                X[:n], inverse=inverse, allow_transfer=batch_size is not None
            ).to(device)
        if out is None:
            destination = torch.empty(
                (n, width), device=device, dtype=self.components_.dtype
            )
            out = destination
        else:
            flags = getattr(out, "flags", None)
            if flags is not None and not flags.writeable:
                raise ValueError("out must be writable.")
            destination = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
            if (
                destination.shape != (n, width)
                or destination.dtype != self.components_.dtype
            ):
                raise ValueError(
                    f"out must have shape {(n, width)} "
                    f"and dtype {self.components_.dtype}."
                )
            if output_device is not None and destination.device != device:
                raise ValueError("out and output_device must use the same device.")
        for batch in self.gen_batches(n, size):
            destination[batch].copy_(
                self._project_batch(
                    X[batch], inverse=inverse, allow_transfer=batch_size is not None
                )
            )
        return out

    def transform(self, X, *, batch_size=None, output_device=None, out=None):
        """Project in batches into one tensor or a supplied tensor/NumPy buffer.

        Results default to the model device. ``output_device`` changes the
        destination; ``out`` may also be a writable NumPy memmap. Use
        ``transform_batches`` to avoid allocating the complete output.
        The fitted mean is subtracted before multiplication for stability.
        """
        return self._transform_into(X, batch_size, output_device, out)

    def inverse_transform(self, X, *, batch_size=None, output_device=None, out=None):
        """Reconstruct features, undoing whitening first when enabled."""
        return self._transform_into(X, batch_size, output_device, out, inverse=True)

    def fit_transform(
        self, X, check_input=True, *, batch_size=None, output_device=None, out=None
    ):
        """Fit, then replay X using the final basis. X must be replayable.

        The fitting pass preserves X even with ``copy=False``, so the second
        pass sees the original data without needing a full dataset copy.
        """
        copy = self.copy
        try:
            self.copy = True
            self.fit(X, check_input=check_input)
        finally:
            self.copy = copy
        return self.transform(
            X, batch_size=batch_size, output_device=output_device, out=out
        )

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
