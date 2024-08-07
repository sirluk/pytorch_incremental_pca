# PyTorch Incremental PCA

[![PyPI Version](https://img.shields.io/pypi/v/torch-incremental-pca.svg)](https://pypi.org/project/torch-incremental-pca/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This project provides a PyTorch implementation of the Incremental PCA algorithm, inspired by the `IncrementalPCA` class from scikit-learn and the repository [PCAonGPU](https://github.com/dnhkng/PCAonGPU/tree/main). This implementation has some shortcomings with regards to the precision of operations leading to vastly different results compared to the sklearn implemtation. The IncrementalPCA class in this repo produces outputs which are very close to the sklearn implementation with the added benefit of running on GPU.

Incremental PCA is a valuable technique for dimensionality reduction when dealing with large datasets that cannot fit entirely into memory.

## Features

* **PyTorch Integration:** Seamlessly use incremental PCA within your PyTorch workflows.
* **Memory Efficiency:**  Process large datasets incrementally without loading everything into memory at once.
* **Similar API:**  Familiar interface if you've used scikit-learn's `IncrementalPCA`.
* **Customization:** Easily extend or modify the core functionality to suit your specific needs.

## Installation

```bash
pip install torch-incremental-pca
```

## Usage
```python
import torch_incremental_pca as tip

pca = tip.IncrementalPCA(n_components=32)
```
