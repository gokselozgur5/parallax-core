"""Embedding backward weight gradient accumulation kernels."""

from __future__ import annotations

import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


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



def sparse_embedding_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    padding_idx: int | None = None,
    num_embeddings: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sparse unique index gradient representation."""
    if num_embeddings is not None and num_embeddings <= 0:
        raise ValueError("num_embeddings must be a positive integer")
    idx_arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(idx_arr < 0) or (num_embeddings is not None and np.any(idx_arr >= num_embeddings)):
        raise IndexError("indices out of range for num_embeddings")
    grad_arr = np.asarray(grad_output).reshape(idx_arr.shape[0], -1)
    unique_idx, inv = np.unique(idx_arr, return_inverse=True)
    accum_grad = np.zeros((len(unique_idx), grad_arr.shape[1]), dtype=grad_arr.dtype)
    np.add.at(accum_grad, inv, grad_arr)
    if padding_idx is not None:
        if padding_idx < 0 or (num_embeddings is not None and padding_idx >= num_embeddings):
            raise IndexError("padding_idx out of range for num_embeddings")
        mask = unique_idx == padding_idx
        if np.any(mask):
            accum_grad[mask] = 0.0
    return unique_idx, accum_grad



def accumulate_embedding_backward(
    grad_output: Tensor | np.ndarray,
    indices: Tensor | np.ndarray,
    weight: Tensor,
    padding_idx: int | None = None,
) -> None:
    """Accumulate backward gradients into weight Tensor with autograd support."""
    if not getattr(weight, "requires_grad", True):
        return
    grad_arr = grad_output.data if isinstance(grad_output, Tensor) else np.asarray(grad_output)
    idx_arr = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
    num_emb, emb_dim = weight.shape[:2]
    dw = embedding_backward(grad_arr, idx_arr, num_emb, emb_dim, padding_idx=padding_idx)
    _accumulate_grad(weight, dw)



class DispatchedEmbeddingBackward:
    """Dispatched embedding backward weight gradient accumulator."""

    compute = staticmethod(embedding_backward)
    accumulate = staticmethod(accumulate_embedding_backward)
    sparse_compute = staticmethod(sparse_embedding_backward)

    def __repr__(self) -> str:
        return "DispatchedEmbeddingBackward()"



__all__ = [
    "DispatchedEmbeddingBackward",
    "accumulate_embedding_backward",
    "embedding_backward",
    "sparse_embedding_backward",
]
