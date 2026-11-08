"""Odds Ratio Preference Optimization (ORPO) without reference model.

Reference: Hong et al., 2024 - ORPO: Monolithic Preference Optimization without
Reference Model.
Combines Supervised Fine-Tuning (SFT) negative log-likelihood with an Odds Ratio (OR)
penalty between winning and losing sequences.
"""

from typing import Tuple
import numpy as np


def compute_log_odds(log_probs: np.ndarray) -> float:
    """Compute log odds log(P / (1 - P)) from sequence average log probability.

    Args:
        log_probs: Token log probabilities for sequence, shape (seq_len,).

    Returns:
        Log odds value (scalar float).
    """
    avg_log_p = float(np.mean(log_probs))
    # Clip to avoid numerical instability
    avg_log_p = np.clip(avg_log_p, -30.0, -1e-6)
    p = np.exp(avg_log_p)
    # log(p / (1 - p)) = log(p) - log(1 - p)
    log_one_minus_p = np.log1p(-p + 1e-12)
    return float(avg_log_p - log_one_minus_p)
