"""
Numerically stable loss functions for language modeling.

Implements categorical cross-entropy loss with the log-sum-exp trick
to prevent numerical underflow/overflow.
"""

from typing import Tuple
import numpy as np
from parallax.tensor import Tensor, unbroadcast_grad


def cross_entropy_loss(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Computes mean categorical cross-entropy loss.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size] or [N, vocab_size]
        targets: ndarray of integer labels with shape [batch_size, seq_len] or [N]

    Returns:
        Scalar Tensor representing the mean cross-entropy loss.
    """
    logits_data = logits.data
    orig_shape = logits_data.shape
    V = orig_shape[-1]
    flat_logits = logits_data.reshape(-1, V)
    flat_targets = targets.reshape(-1)
    N = flat_logits.shape[0]

    # Log-sum-exp trick: log(sum(exp(x - max(x)))) + max(x)
    max_logits = np.max(flat_logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(flat_logits - max_logits)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = (flat_logits - max_logits) - np.log(sum_exp)
    probs = exp_shifted / sum_exp

    # Loss is negative log likelihood of correct targets: -log_probs[i, target[i]]
    nll = -log_probs[np.arange(N), flat_targets]
    mean_loss = float(np.mean(nll))

    out = Tensor(mean_loss, (logits,), "CrossEntropyLoss")

    def _backward() -> None:
        if logits.requires_grad and logits.grad is not None:
            # Gradient of softmax cross entropy: (probs - 1_target) / N
            dlogits = probs.copy()
            dlogits[np.arange(N), flat_targets] -= 1.0
            dlogits = (dlogits / N) * float(out.grad if out.grad is not None else 1.0)
            logits.grad += unbroadcast_grad(dlogits.reshape(orig_shape), logits.shape)

    out._backward = _backward
    return out
