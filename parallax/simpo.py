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


def simpo_step(
    logits_w: np.ndarray,
    targets_w: np.ndarray,
    logits_l: np.ndarray,
    targets_l: np.ndarray,
    beta: float = 2.0,
    gamma_margin: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute SimPO loss and analytical gradients for chosen and rejected logits.

    Args:
        logits_w: Chosen sequence logits (seq_len_w, vocab_size).
        targets_w: Chosen target tokens (seq_len_w,).
        logits_l: Rejected sequence logits (seq_len_l, vocab_size).
        targets_l: Rejected target tokens (seq_len_l,).
        beta: Reward temperature.
        gamma_margin: Target separation margin.

    Returns:
        Tuple of (loss, grad_logits_w, grad_logits_l).
    """
    def get_token_log_probs(logits, targets):
        m = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - m)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        seq_len = len(targets)
        log_probs = np.zeros(seq_len)
        for i in range(seq_len):
            log_probs[i] = np.log(probs[i, targets[i]] + 1e-12)
        return log_probs, probs

    lp_w, probs_w = get_token_log_probs(logits_w, targets_w)
    lp_l, probs_l = get_token_log_probs(logits_l, targets_l)

    loss = compute_simpo_loss(lp_w, lp_l, beta=beta, gamma_margin=gamma_margin)

    # Derivative dL/ddiff = -sigma(-diff)
    r_w = compute_simpo_reward(lp_w, beta=beta)
    r_l = compute_simpo_reward(lp_l, beta=beta)
    diff = r_w - r_l - gamma_margin
    scalar_grad = -1.0 / (1.0 + np.exp(np.clip(diff, -30.0, 30.0)))

    # Chosen gradient
    len_w = len(targets_w)
    grad_w = np.zeros_like(logits_w)
    scale_w = (beta / len_w) * scalar_grad
    for i in range(len_w):
        grad_w[i] = probs_w[i] * (-scale_w)
        grad_w[i, targets_w[i]] -= -scale_w

    # Rejected gradient
    len_l = len(targets_l)
    grad_l = np.zeros_like(logits_l)
    scale_l = -(beta / len_l) * scalar_grad
    for i in range(len_l):
        grad_l[i] = probs_l[i] * (-scale_l)
        grad_l[i, targets_l[i]] -= -scale_l

    return loss, grad_w, grad_l
