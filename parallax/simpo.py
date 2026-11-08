"""Simple Preference Optimization (SimPO) with Target-Length Normalization.

Reference: Meng et al., 2024 - SimPO: Simple Preference Optimization with a
Reference-Free Objective.
Uses average sequence log-probabilities as implicit rewards, eliminating length bias,
and enforces a target reward margin gamma between chosen and rejected generations.
"""

from typing import Tuple
import numpy as np


def compute_simpo_reward(log_probs: np.ndarray, beta: float = 2.0) -> float:
    """Compute length-normalized sequence reward: r(x, y) = (beta / |y|) * sum(log_probs).

    Args:
        log_probs: Token log probabilities of shape (seq_len,).
        beta: Scaling temperature hyperparameter.

    Returns:
        Scalar reward float.
    """
    seq_len = len(log_probs)
    if seq_len == 0:
        return 0.0
    return float(beta * np.mean(log_probs))


def compute_simpo_loss(
    log_probs_w: np.ndarray,
    log_probs_l: np.ndarray,
    beta: float = 2.0,
    gamma_margin: float = 0.5,
) -> float:
    """Compute SimPO loss: -log sigma(r(x, y_w) - r(x, y_l) - gamma).

    Args:
        log_probs_w: Chosen sequence log probs (seq_len_w,).
        log_probs_l: Rejected sequence log probs (seq_len_l,).
        beta: Reward scaling factor.
        gamma_margin: Target reward margin between winner and loser.

    Returns:
        Scalar loss value.
    """
    r_w = compute_simpo_reward(log_probs_w, beta=beta)
    r_l = compute_simpo_reward(log_probs_l, beta=beta)
    diff = r_w - r_l - gamma_margin
    # -log sigma(x) = log(1 + exp(-x))
    loss = float(np.log1p(np.exp(-np.clip(diff, -30.0, 30.0))))
    return loss
