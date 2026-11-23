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
    targets_flat = targets_flat.astype(np.int64)
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
    targets_flat = targets_flat.astype(np.int64)
    probs = np.exp(logits_flat - lse.reshape(N, 1))
    grad = np.zeros_like(logits_flat)
    for i in range(N):
        if not valid[i]:
            continue
        tgt = targets_flat[i]
        if label_smoothing > 0.0:
            target_dist = np.full(C, label_smoothing / C, dtype=logits.dtype)
            target_dist[tgt] += (1.0 - label_smoothing)
            grad[i] = probs[i] - target_dist
        else:
            grad[i] = probs[i].copy()
            grad[i, tgt] -= 1.0
    if reduction == "mean":
        scale = grad_out / max(np.sum(valid), 1)
        grad = grad * scale
    elif reduction == "sum":
        grad = grad * grad_out
    else:
        grad = grad * np.asarray(grad_out).reshape(N, 1)
    return grad.reshape(logits.shape)


def _dispatch_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route cross entropy calculation through dispatch engine or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch(
            "cross_entropy", logits, targets,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    except Exception:
        return cross_entropy_forward_reference(
            logits, targets, ignore_index, label_smoothing, reduction, chunk_size
        )


def fast_cross_entropy(
    logits: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    chunk_size: Optional[int] = None,
) -> Tensor:
    """Dispatched fast Cross-Entropy loss with autograd backward hook."""
    raw_logits = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    raw_targets = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    loss_val, lse, _ = _dispatch_cross_entropy(
        raw_logits, raw_targets, ignore_index, label_smoothing, reduction, chunk_size
    )
    parents = (logits,) if isinstance(logits, Tensor) else ()
    out = Tensor(loss_val, parents, "fast_cross_entropy")
    if not out.requires_grad:
        return out
    def _backward() -> None:
        if out.grad is None:
            return
        d_logits = cross_entropy_backward_reference(
            out.grad, raw_logits, raw_targets, lse,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
        if isinstance(logits, Tensor) and logits.requires_grad:
            _accumulate_grad(logits, d_logits)

    out._backward = _backward
    return out


class DispatchedCrossEntropyLoss(Module):
    """Dispatched fast cross-entropy loss module with online log-sum-exp."""

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        self.reduction = str(reduction)
        self.chunk_size = chunk_size

    def forward(self, logits: Tensor | np.ndarray, targets: Tensor | np.ndarray) -> Tensor:
        return fast_cross_entropy(
            logits, targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
            chunk_size=self.chunk_size,
        )
    def extra_repr(self) -> str:
        return (
            f"ignore_index={self.ignore_index}, "
            f"label_smoothing={self.label_smoothing}, "
            f"reduction={self.reduction!r}"
        )

    def __repr__(self) -> str:
        return f"DispatchedCrossEntropyLoss({self.extra_repr()})"


CrossEntropyLossModule = DispatchedCrossEntropyLoss


__all__ = [
    "CrossEntropyLossModule",
    "DispatchedCrossEntropyLoss",
    "cross_entropy_backward_reference",
    "cross_entropy_forward_reference",
    "fast_cross_entropy",
    "online_logsumexp",
]
