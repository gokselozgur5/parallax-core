"""Embedding backward weight gradient accumulation kernels."""

from __future__ import annotations

import numpy as np


def embedding_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    num_embeddings: int,
    embedding_dim: int,
) -> np.ndarray:
    """Accumulate incoming gradients into embedding weight gradient tensor."""
    grad_weight = np.zeros((num_embeddings, embedding_dim), dtype=grad_output.dtype)
    flat_indices = np.asarray(indices).reshape(-1)
    flat_grad = np.asarray(grad_output).reshape(-1, embedding_dim)
    np.add.at(grad_weight, flat_indices, flat_grad)
    return grad_weight
