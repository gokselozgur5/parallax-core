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
