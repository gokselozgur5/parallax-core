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
