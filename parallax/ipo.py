"""Identity Preference Optimization (IPO) direct alignment kernel."""

from typing import Tuple
import numpy as np


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
) -> np.ndarray:
    """Compute gradients of policy logits with respect to IPO objective."""
    # Chosen receives negative grad (increase prob), rejected receives positive (decrease prob)
    grad_logits = np.zeros_like(policy_logits)
    batch_size = len(chosen_labels)
    for b in range(batch_size):
        w = grad_weight[b]
        grad_logits[b, chosen_labels[b]] -= w
        grad_logits[b, rejected_labels[b]] += w
    return grad_logits
