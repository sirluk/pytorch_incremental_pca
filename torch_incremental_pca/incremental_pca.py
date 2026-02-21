from typing import Optional, Tuple

import torch


class IncrementalPCA:
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.
    Adapted from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py

    This class provides methods to fit the model on data incrementally in batches, and to transform new data based on
    the principal components learned during the fitting process.

    Three SVD backends are available, selectable via the `lowrank` and `gram` flags:

    - **Full SVD** (default): Uses `torch.linalg.svd`. Exact but uses Householder
      bidiagonalization which is inherently sequential and underutilizes GPU parallelism.
    - **Low-rank SVD** (`lowrank=True`): Uses `torch.svd_lowrank` (randomized SVD).
      Faster for very large matrices where `n_components << min(n_samples, n_features)`,
      but involves multiple power iterations that can be slower for moderate-sized matrices.
    - **Gram eigendecomposition** (`gram=True`): Computes `G = X @ X.T` (a GEMM) then
      `torch.linalg.eigh(G)` to recover singular values/vectors. Mathematically equivalent
      to full SVD but significantly faster on GPU because it replaces sequential Householder
      reflections with highly-parallelizable matrix multiplications. Recommended when the
      augmented matrix is wide (rows < cols), which is the typical case in incremental PCA
      when `n_components + batch_size < n_features`.

    Args:
        n_components (int, optional): Number of components to keep. If `None`, it's set to the minimum of the
            number of samples and features. Defaults to None.
        copy (bool): If False, input data will be overwritten. Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch. Only needed if self.fit is called.
            If `None`, it's inferred from the data and set to `5 * n_features`. Defaults to None.
        svd_driver (str, optional): name of the cuSOLVER method to be used for torch.linalg.svd. This keyword
            argument only works on CUDA inputs. Available options are: None, gesvd, gesvdj, and gesvda. Defaults to
            None.
        lowrank (bool, optional): Whether to use torch.svd_lowrank instead of torch.linalg.svd which can be faster.
            Mutually exclusive with `gram`. Defaults to False.
        lowrank_q (int, optional): For an adequate approximation of n_components, this parameter defaults to
            n_components * 2.
        lowrank_niter (int, optional): Number of subspace iterations to conduct for torch.svd_lowrank.
            Defaults to 4.
        lowrank_seed (int, optional): Seed for making results of torch.svd_lowrank reproducible.
        gram (bool, optional): Whether to use gram-matrix eigendecomposition instead of
            torch.linalg.svd. For wide matrices (rows < cols), this computes G = X @ X.T
            followed by torch.linalg.eigh(G) and recovers singular vectors via
            Vt = U.T @ X / S. Mathematically equivalent to full SVD but significantly faster
            on GPU because it uses GEMM operations instead of sequential Householder
            reflections. Falls back to full SVD when the matrix is tall (rows > cols).
            Mutually exclusive with `lowrank`. Defaults to False.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        copy: Optional[bool] = True,
        batch_size: Optional[int] = None,
        svd_driver: Optional[str] = None,
        lowrank: bool = False,
        lowrank_q: Optional[int] = None,
        lowrank_niter: int = 4,
        lowrank_seed: Optional[int] = None,
        gram: bool = False,
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

        self.n_features_ = None

        if self.lowrank and self.gram:
            raise ValueError("lowrank and gram are mutually exclusive. Set only one to True.")

        if self.lowrank:
            self._validate_lowrank_params()

    def _validate_lowrank_params(self):
        if self.lowrank_q is None:
            if self.n_components is None:
                raise ValueError("n_components must be specified when using lowrank mode with lowrank_q=None.")
            self.lowrank_q = self.n_components * 2
        elif self.lowrank_q < self.n_components:
            raise ValueError("lowrank_q must be greater than or equal to n_components.")

    def _svd_fn_full(self, X):
        return torch.linalg.svd(X, full_matrices=False, driver=self.svd_driver)

    def _svd_fn_lowrank(self, X):
        seed_enabled = self.lowrank_seed is not None
        with torch.random.fork_rng(enabled=seed_enabled):
            if seed_enabled:
                torch.manual_seed(self.lowrank_seed)
            U, S, V = torch.svd_lowrank(X, q=self.lowrank_q, niter=self.lowrank_niter)
            return U, S, V.mH

    def _svd_fn_gram(self, X):
        """Compute SVD via gram-matrix eigendecomposition.

        For wide matrices (m < D), instead of running the sequential Householder
        bidiagonalization used by torch.linalg.svd, this method computes the
        small gram matrix G = X @ X.T (shape (m, m)) using a single GEMM,
        then finds its eigendecomposition via torch.linalg.eigh. The right
        singular vectors are recovered as Vt = diag(1/S) @ U.T @ X, which is
        another GEMM.

        This is mathematically equivalent to full SVD but significantly faster on
        GPU because GEMM operations achieve much higher hardware utilization than
        the sequential Householder reflections used by torch.linalg.svd.

        Falls back to full SVD when the matrix is tall (m > D).

        Args:
            X (torch.Tensor): Input matrix of shape (m, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (U, S, Vt) where
                U is (m, m), S is (m,), Vt is (m, D).
        """
        m, D = X.shape
        if m > D:
            return self._svd_fn_full(X)
        G = X @ X.mT  # (m, m) — GEMM
        evals, evecs = torch.linalg.eigh(G)  # ascending order
        # Flip to descending order (largest eigenvalues first)
        evals = evals.flip(0)
        evecs = evecs.flip(1)
        S = torch.sqrt(evals.clamp(min=0))
        # Recover right singular vectors: Vt = diag(1/S) @ U.T @ X
        valid = S > 1e-7
        Vt = torch.zeros(m, D, dtype=X.dtype, device=X.device)
        Vt[valid] = (evecs[:, valid].mT @ X) / S[valid].unsqueeze(1)
        return evecs, S, Vt

    def _validate_data(self, X) -> torch.Tensor:
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        Args:
            X (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Converted to appropriate format.
        """
        valid_dtypes = [torch.float32, torch.float64]

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        elif self.copy:
            X = X.clone()

        n_samples, n_features = X.shape
        if self.n_components is None:
            pass
        elif self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} invalid for n_features={n_features}, "
                "need more rows than columns for IncrementalPCA processing."
            )
        elif self.n_components > n_samples:
            raise ValueError(
                f"n_components={self.n_components} must be less or equal to the batch number of samples {n_samples}"
            )

        if X.dtype not in valid_dtypes:
            X = X.to(torch.float32)

        return X

    @staticmethod
    def _incremental_mean_and_var(
        X, last_mean, last_variance, last_sample_count
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the incremental mean and variance for the data `X`.

        Uses float64 internally for numerical stability when accumulating sums,
        matching the behavior of scikit-learn's implementation. The results are
        cast back to the input dtype before returning to avoid propagating float64
        into downstream computations (e.g., SVD), which would significantly hurt
        GPU performance.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        if last_sample_count > 0:
            if last_mean is None:
                raise ValueError("last_mean should not be None if last_sample_count > 0.")
            if last_variance is None:
                raise ValueError("last_variance should not be None if last_sample_count > 0.")

        input_dtype = X.dtype

        new_sample_count = torch.tensor([X.shape[0]], device=X.device)
        updated_sample_count = last_sample_count + new_sample_count

        if last_mean is None:
            last_sum = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
        else:
            last_sum = last_mean.double() * last_sample_count

        new_sum = X.sum(dim=0, dtype=torch.float64)

        updated_mean = (last_sum + new_sum) / updated_sample_count

        T = new_sum / new_sample_count
        temp = X - T
        correction = temp.sum(dim=0, dtype=torch.float64).square()
        temp.square_()
        new_unnormalized_variance = temp.sum(dim=0, dtype=torch.float64)
        new_unnormalized_variance -= correction / new_sample_count
        if last_variance is None:
            updated_variance = new_unnormalized_variance / updated_sample_count
        else:
            last_unnormalized_variance = last_variance.double() * last_sample_count
            last_over_new_count = last_sample_count.double() / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count / updated_sample_count * (last_sum / last_over_new_count - new_sum).square()
            )
            updated_variance = updated_unnormalized_variance / updated_sample_count

        # Cast back to input dtype to avoid float64 propagating into SVD/eigendecomp,
        # which would severely hurt GPU performance (float64 throughput is up to 32x
        # lower than float32 on consumer GPUs).
        return updated_mean.to(input_dtype), updated_variance.to(input_dtype), updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping.
                Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs[: u.shape[1]].view(1, -1)
        v *= signs.view(-1, 1)
        return u, v

    def fit(self, X, check_input=True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size = 5 * n_features

        for batch in self.gen_batches(n_samples, self.batch_size, min_batch_size=self.n_components or 0):
            self.partial_fit(X[batch], check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCA: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        # Initialize attributes to avoid errors during the first call to partial_fit
        if first_pass:
            self.mean_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.var_ = None  # Will be initialized properly in _incremental_mean_and_var based on data dimensions
            self.n_samples_seen_ = torch.tensor([0], device=X.device)
            self.n_features_ = n_features
            if not self.n_components:
                self.n_components = min(n_samples, n_features)

        if n_features != self.n_features_:
            raise ValueError(
                "Number of features of the new batch does not match the number of features of the first batch."
            )

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, self.n_samples_seen_
        )

        if first_pass:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt((self.n_samples_seen_.to(X.dtype) / n_total_samples) * n_samples)
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.view((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        if self.lowrank:
            U, S, Vt = self._svd_fn_lowrank(X)
        elif self.gram:
            U, S, Vt = self._svd_fn_gram(X)
        else:
            U, S, Vt = self._svd_fn_full(X)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components]
        self.singular_values_ = S[: self.n_components]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components]
        if self.n_components not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        return self

    def transform(self, X) -> torch.Tensor:
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X = X - self.mean_
        return torch.mm(X, self.components_.T)

    @staticmethod
    def gen_batches(n: int, batch_size: int, min_batch_size: int = 0):
        """Generator to create slices containing `batch_size` elements from 0 to `n`.

        The last slice may contain less than `batch_size` elements, when `batch_size` does not divide `n`.

        Args:
            n (int): Size of the sequence.
            batch_size (int): Number of elements in each batch.
            min_batch_size (int, optional): Minimum number of elements in each batch. Defaults to 0.

        Yields:
            slice: A slice of `batch_size` elements.
        """
        start = 0
        for _ in range(int(n // batch_size)):
            end = start + batch_size
            if end + min_batch_size > n:
                continue
            yield slice(start, end)
            start = end
        if start < n:
            yield slice(start, n)
