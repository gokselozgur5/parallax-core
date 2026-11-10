"""
Numerically stable loss functions and evaluation metrics for language modeling.

Implements:
- Categorical cross-entropy loss with log-sum-exp trick and label smoothing
- Perplexity (PPL) calculation
"""

import math
import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


def cross_entropy_loss(
    logits: Tensor,
    targets: np.ndarray,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> Tensor:
    """Computes mean categorical cross-entropy loss with optional label smoothing.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size] or [N, vocab_size]
        targets: ndarray of integer labels with shape [batch_size, seq_len] or [N]
        label_smoothing: Float in [0.0, 1.0] for regularization
        ignore_index: Target index to ignore in loss computation (e.g. padding)

    Returns:
        Scalar Tensor representing the mean cross-entropy loss.
    """
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("label_smoothing must be in [0.0, 1.0)")
    logits_data = logits.data
    orig_shape = logits_data.shape
    V = orig_shape[-1]
    flat_logits = logits_data.reshape(-1, V)
    flat_targets = targets.reshape(-1)

    # Validate targets outside [0, V) unless equal to ignore_index
    invalid_targets = (flat_targets != ignore_index) & ((flat_targets < 0) | (flat_targets >= V))
    if np.any(invalid_targets):
        first_bad = int(flat_targets[invalid_targets][0])
        raise ValueError(f"Target index {first_bad} out of bounds for vocab_size {V}")

    # Valid mask for ignore_index
    valid_mask = flat_targets != ignore_index
    N_valid = int(np.sum(valid_mask))
    if N_valid == 0:
        return Tensor(0.0, (logits,), "CrossEntropyLoss")

    # Log-sum-exp trick: log(sum(exp(x - max(x)))) + max(x)
    max_logits = np.max(flat_logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(flat_logits - max_logits)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = (flat_logits - max_logits) - np.log(sum_exp)
    probs = exp_shifted / sum_exp

    indices = np.arange(flat_logits.shape[0])

    if label_smoothing == 0.0:
        nll = -log_probs[indices[valid_mask], flat_targets[valid_mask]]
        mean_loss = float(np.sum(nll) / N_valid)
    else:
        # Smoothed targets: (1 - eps) * one_hot + eps / V
        smooth_nll = -np.sum(log_probs[valid_mask] * (label_smoothing / V), axis=-1)
        target_nll = -log_probs[indices[valid_mask], flat_targets[valid_mask]] * (1.0 - label_smoothing)
        mean_loss = float(np.sum(smooth_nll + target_nll) / N_valid)

    out = Tensor(mean_loss, (logits,), "CrossEntropyLoss")

    def _backward() -> None:
        if logits.requires_grad:
            # Gradient calculation
            dlogits = np.zeros_like(flat_logits)
            if label_smoothing == 0.0:
                dlogits[valid_mask] = probs[valid_mask].copy()
                dlogits[indices[valid_mask], flat_targets[valid_mask]] -= 1.0
            else:
                target_dist = np.full_like(flat_logits[valid_mask], label_smoothing / V)
                target_dist[np.arange(N_valid), flat_targets[valid_mask]] += (1.0 - label_smoothing)
                dlogits[valid_mask] = probs[valid_mask] - target_dist

            loss_grad = float(out.grad if out.grad is not None else 1.0)
            dlogits = (dlogits / N_valid) * loss_grad
            _accumulate_grad(logits, dlogits.reshape(orig_shape))

    out._backward = _backward
    return out


def perplexity_from_loss(loss_value: float) -> float:
    """Computes perplexity (PPL) from cross-entropy loss in natural log scale.

    PPL = exp(CrossEntropyLoss)
    """
    try:
        return float(math.exp(loss_value))
    except OverflowError:
        return float("inf")


# Backward compatibility alias
compute_perplexity = perplexity_from_loss

