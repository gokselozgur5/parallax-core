"""Identity Preference Optimization (IPO) direct alignment kernel."""

from typing import Optional, Tuple, Union
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


class IPOConfig:
    """Configuration container for Identity Preference Optimization."""

    def __init__(self, tau: float = 0.1) -> None:
        self.tau = float(tau)
        self.target_margin = 1.0 / (2.0 * self.tau)

def compute_ipo_loss(
    pi_w_logprob: np.ndarray,
    pi_l_logprob: np.ndarray,
    ref_w_logprob: np.ndarray,
    ref_l_logprob: np.ndarray,
    tau: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """Compute IPO quadratic preference loss and scalar gradient weights."""
    h_w = pi_w_logprob - ref_w_logprob
    h_l = pi_l_logprob - ref_l_logprob
    log_ratio_diff = tau * (h_w - h_l)
    diff = log_ratio_diff - (1.0 / (2.0 * tau))
    loss = float(np.mean(diff ** 2))
    # Gradient multiplier dLoss / d(log_ratio_diff)
    grad_weight = 2.0 * diff * tau / len(pi_w_logprob)
    return loss, grad_weight

def ipo_step(
    policy_logits: np.ndarray,
    chosen_labels: np.ndarray,
    rejected_labels: np.ndarray,
    grad_weight: np.ndarray,
    rejected_logits: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute gradients of policy logits with respect to IPO objective."""
    probs_w = _softmax(policy_logits)
    probs_l = _softmax(rejected_logits) if rejected_logits is not None else probs_w
    grad_w = np.zeros_like(policy_logits)
    grad_l = np.zeros_like(rejected_logits if rejected_logits is not None else policy_logits)
    batch_size = len(chosen_labels)
    for b in range(batch_size):
        w = grad_weight[b]
        # Chosen log-prob gradient: +w * (1_chosen - probs_w)
        grad_chosen = -w * probs_w[b]
        grad_chosen[chosen_labels[b]] += w
        # Rejected log-prob gradient: -w * (1_rejected - probs_l)
        grad_rejected = w * probs_l[b]
        grad_rejected[rejected_labels[b]] -= w
        grad_w[b] = grad_chosen
        grad_l[b] = grad_rejected
    if rejected_logits is not None:
        return grad_w, grad_l
    return grad_w + grad_l
