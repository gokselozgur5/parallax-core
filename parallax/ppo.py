"""Generalized Advantage Estimation (GAE) and Proximal Policy Optimization (PPO).

Reference: Schulman et al., 2015/2017 - High-Dimensional Continuous Control Using
Generalized Advantage Estimation & Proximal Policy Optimization Algorithms.
Computes exponentially-discounted TD residuals and clipped surrogate policy objective.
"""

from typing import Tuple
import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    Args:
        rewards: Step rewards of shape (T,).
        values: Value estimates of shape (T + 1,).
        gamma: Discount factor.
        gae_lambda: Bias-variance tradeoff parameter.

    Returns:
        Tuple of:
            - advantages: GAE advantage estimates of shape (T,).
            - returns: Target empirical returns V(s) + A of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:T]
    return advantages, returns


def compute_ppo_clip_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """Compute PPO clipped surrogate policy loss: -min(r * A, clip(r, 1-eps, 1+eps) * A).

    Args:
        log_probs: New policy token log probabilities (T,).
        old_log_probs: Old rollout policy token log probabilities (T,).
        advantages: Normalized advantage estimates (T,).
        clip_eps: Clipping threshold epsilon (e.g. 0.2).

    Returns:
        Tuple of (mean_clipped_loss, policy_ratios).
    """
    # Ratio r_t(theta) = exp(log pi_new - log pi_old)
    ratios = np.exp(np.clip(log_probs - old_log_probs, -20.0, 20.0))
    surr1 = ratios * advantages
    clipped_ratios = np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratios * advantages

    # PPO maximizes min(surr1, surr2), so loss is -min(surr1, surr2)
    loss = -float(np.mean(np.minimum(surr1, surr2)))
    return loss, ratios


def ppo_policy_step(
    logits: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float = 0.2,
) -> Tuple[float, np.ndarray]:
    """Perform a PPO clipped policy gradient step, computing loss and dL/dlogits.

    Args:
        logits: Policy logits of shape (T, vocab_size).
        actions: Taken token action IDs of shape (T,).
        old_log_probs: Rollout policy token log probs of shape (T,).
        advantages: Advantage estimates of shape (T,).
        clip_eps: Clipping margin.

    Returns:
        Tuple of (loss, grad_logits).
    """
    T, vocab_size = logits.shape
    # Log-softmax and probabilities
    m = np.max(logits, axis=-1, keepdims=True)
    logsumexp = m + np.log(np.sum(np.exp(logits - m), axis=-1, keepdims=True))
    log_probs_all = logits - logsumexp
    probs = np.exp(log_probs_all)

    log_probs = np.zeros(T, dtype=np.float64)
    for t in range(T):
        log_probs[t] = log_probs_all[t, actions[t]]

    # Normalize advantages
    if len(advantages) > 1 and np.std(advantages) > 1e-8:
        adv_norm = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    else:
        adv_norm = np.array(advantages, dtype=np.float64)

    loss, ratios = compute_ppo_clip_loss(log_probs, old_log_probs, adv_norm, clip_eps)

    grad_logits = np.zeros_like(logits)
    for t in range(T):
        r = ratios[t]
        adv = adv_norm[t]
        # Active gradient only if not clipped
        if (adv >= 0 and r <= 1.0 + clip_eps) or (adv < 0 and r >= 1.0 - clip_eps):
            scale = -r * adv / T
            # For gradient descent: dLoss/dlogits = scale * (1_action - probs)
            grad_logits[t] = -scale * probs[t]
            grad_logits[t, actions[t]] += scale

    return loss, grad_logits
