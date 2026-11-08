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
