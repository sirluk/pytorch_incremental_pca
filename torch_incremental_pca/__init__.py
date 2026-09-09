"""Incremental PCA on GPU and CPU with PyTorch.

Provides :class:`IncrementalPCA`, a scikit-learn-style incremental principal
component analysis estimator for CUDA and CPU tensors, NumPy arrays, and memory
maps, for out-of-core dimensionality reduction of datasets larger than memory.
"""

from torch_incremental_pca.incremental_pca import IncrementalPCA  # noqa: F401
