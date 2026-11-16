"""Embedding backward weight gradient accumulation kernels."""

from __future__ import annotations

import numpy as np


def embedding_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate incoming gradients into embedding weight gradient tensor."""
    if num_embeddings <= 0 or embedding_dim <= 0:
        raise ValueError("num_embeddings and embedding_dim must be positive integers")
    idx_arr = np.asarray(indices, dtype=np.int64)
    grad_arr = np.asarray(grad_output)
    if grad_arr.shape[-1] != embedding_dim or grad_arr.shape[:-1] != idx_arr.shape:
        raise ValueError(f"grad_output shape {grad_arr.shape} incompatible with indices {idx_arr.shape}")
    if np.any(idx_arr < 0) or np.any(idx_arr >= num_embeddings):
        raise IndexError("indices out of range for num_embeddings")
    if out is not None:
        if out.shape != (num_embeddings, embedding_dim):
            raise ValueError(f"out shape {out.shape} != {(num_embeddings, embedding_dim)}")
        grad_weight = out
    else:
        grad_weight = np.zeros((num_embeddings, embedding_dim), dtype=grad_arr.dtype)
    flat_indices = idx_arr.reshape(-1)
    flat_grad = grad_arr.reshape(-1, embedding_dim)
    np.add.at(grad_weight, flat_indices, flat_grad)
    if padding_idx is not None:
        if padding_idx < 0 or padding_idx >= num_embeddings:
            raise IndexError("padding_idx out of range for num_embeddings")
        grad_weight[padding_idx] = 0.0
    return grad_weight
