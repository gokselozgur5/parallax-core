"""Dispatched Fast Cross-Entropy loss with online log-sum-exp stabilization."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def online_logsumexp(
    x: np.ndarray,
    chunk_size: Optional[int] = None,
    axis: int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-sum-exp stably using online streaming max and sum-exp."""
    if chunk_size is None or chunk_size >= x.shape[axis]:
        m = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - m)
        s = np.sum(exp_x, axis=axis, keepdims=True)
        lse = m + np.log(s)
        return lse, m, s
    dim = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = 1
    m = np.full(shape, -np.inf, dtype=x.dtype)
    s = np.zeros(shape, dtype=x.dtype)
    for start in range(0, dim, chunk_size):
        end = min(start + chunk_size, dim)
        chunk = np.take(x, indices=range(start, end), axis=axis)
        chunk_m = np.max(chunk, axis=axis, keepdims=True)
        new_m = np.maximum(m, chunk_m)
        s = s * np.exp(m - new_m) + np.sum(
            np.exp(chunk - new_m), axis=axis, keepdims=True
        )
        m = new_m
    lse = m + np.log(s)
    return lse, m, s


def _validate_targets(
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """Validate target shape and value bounds against class count."""
    flat = targets.ravel()
    for val in flat:
        if val != ignore_index and (val < 0 or val >= num_classes):
            raise ValueError(
                f"Target index {val} out of bounds for {num_classes} classes"
            )
    return (flat != ignore_index) & (flat >= 0) & (flat < num_classes)


def cross_entropy_forward_reference(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cross-entropy forward loss with online log-sum-exp."""
    N = targets.size
    C = logits.shape[-1]
    logits_flat = logits.reshape(N, C)
    targets_flat = targets.reshape(N)
    valid = _validate_targets(targets_flat, C, ignore_index)
    lse_flat, _, _ = online_logsumexp(logits_flat, chunk_size=chunk_size, axis=-1)
    lse = lse_flat.reshape(N)
    losses = np.zeros(N, dtype=logits.dtype)
    valid_indices = np.where(valid)[0]
    for i in valid_indices:
        tgt = targets_flat[i]
        nll = lse[i] - logits_flat[i, tgt]
        if label_smoothing > 0.0:
            smooth_loss = lse[i] - np.mean(logits_flat[i])
            losses[i] = (1.0 - label_smoothing) * nll + label_smoothing * smooth_loss
        else:
            losses[i] = nll
    if reduction == "mean":
        total_valid = np.sum(valid)
        loss = np.sum(losses) / max(total_valid, 1) if total_valid > 0 else np.array(0.0, dtype=logits.dtype)
    elif reduction == "sum":
        loss = np.sum(losses)
    else:
        loss = losses.reshape(targets.shape)
    return np.asarray(loss), lse, valid


def cross_entropy_backward_reference(
    grad_out: np.ndarray | float,
    logits: np.ndarray,
    targets: np.ndarray,
    lse: np.ndarray,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> np.ndarray:
    """Compute analytical gradient of cross-entropy with respect to logits."""
    N = targets.size
    C = logits.shape[-1]
    logits_flat = logits.reshape(N, C)
    targets_flat = targets.reshape(N)
    valid = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < C)
    probs = np.exp(logits_flat - lse.reshape(N, 1))
    grad = np.zeros_like(logits_flat)
