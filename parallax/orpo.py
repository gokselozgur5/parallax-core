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


def compute_odds_ratio_loss(
    log_probs_w: np.ndarray,
    log_probs_l: np.ndarray,
    lambda_weight: float = 0.1,
) -> Tuple[float, float, float]:
    """Compute ORPO monolithic loss: L_ORPO = L_SFT + lambda * L_OR.

    Args:
        log_probs_w: Token log probabilities for chosen (winning) response.
        log_probs_l: Token log probabilities for rejected (losing) response.
        lambda_weight: Weight factor for odds ratio penalty.

    Returns:
        Tuple of (total_loss, sft_loss, or_loss).
    """
    sft_loss = -float(np.mean(log_probs_w))

    log_odds_w = compute_log_odds(log_probs_w)
    log_odds_l = compute_log_odds(log_probs_l)

    # log odds ratio: log(odds_w / odds_l) = log_odds_w - log_odds_l
    delta = log_odds_w - log_odds_l

    # L_OR = -log sigma(delta) = log(1 + exp(-delta))
    or_loss = float(np.log1p(np.exp(-np.clip(delta, -30.0, 30.0))))

    total_loss = sft_loss + lambda_weight * or_loss
    return total_loss, sft_loss, or_loss
