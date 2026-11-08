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


def orpo_step(
    logits_w: np.ndarray,
    targets_w: np.ndarray,
    logits_l: np.ndarray,
    targets_l: np.ndarray,
    lambda_weight: float = 0.1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute ORPO loss and analytical gradients with respect to logits_w and logits_l.

    Args:
        logits_w: Chosen sequence logits (seq_len_w, vocab_size).
        targets_w: Chosen target tokens (seq_len_w,).
        logits_l: Rejected sequence logits (seq_len_l, vocab_size).
        targets_l: Rejected target tokens (seq_len_l,).
        lambda_weight: Penalty scaling.

    Returns:
        Tuple of (loss, grad_logits_w, grad_logits_l).
    """
    def get_token_log_probs(logits: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - m)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        probs = exp_logits / sum_exp
        log_probs_all = (logits - m) - np.log(sum_exp)
        log_probs = log_probs_all[np.arange(len(targets)), targets]
        return log_probs, probs

    lp_w, probs_w = get_token_log_probs(logits_w, targets_w)
    lp_l, probs_l = get_token_log_probs(logits_l, targets_l)

    loss, sft_loss, or_loss = compute_odds_ratio_loss(lp_w, lp_l, lambda_weight)

    avg_lp_w = np.clip(float(np.mean(lp_w)), -30.0, -1e-6)
    p_w = np.exp(avg_lp_w)
    d_log_odds_w = 1.0 / (1.0 - p_w + 1e-12)

    avg_lp_l = np.clip(float(np.mean(lp_l)), -30.0, -1e-6)
    p_l = np.exp(avg_lp_l)
    d_log_odds_l = 1.0 / (1.0 - p_l + 1e-12)

    delta = compute_log_odds(lp_w) - compute_log_odds(lp_l)
    d_or_d_delta = -1.0 / (1.0 + np.exp(np.clip(delta, -30.0, 30.0)))

    len_w = len(targets_w)
    len_l = len(targets_l)
    scale_w = (-1.0 + lambda_weight * d_or_d_delta * d_log_odds_w) / len_w
    scale_l = (-lambda_weight * d_or_d_delta * d_log_odds_l) / len_l

    grad_w = np.zeros_like(logits_w)
    for i in range(len_w):
        one_hot = np.zeros(logits_w.shape[-1])
        one_hot[targets_w[i]] = 1.0
        grad_w[i] = scale_w * (one_hot - probs_w[i])

    grad_l = np.zeros_like(logits_l)
    for i in range(len_l):
        one_hot = np.zeros(logits_l.shape[-1])
        one_hot[targets_l[i]] = 1.0
        grad_l[i] = scale_l * (one_hot - probs_l[i])

    return loss, grad_w, grad_l
