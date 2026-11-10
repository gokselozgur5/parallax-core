"""Dynamic Speculative Depth with Shannon entropy early-exit classifier.

Computes predictive confidence (entropy or top-1/top-2 margin) from intermediate
hidden layers and exits early when certainty exceeds a threshold.

Note:
    Architectural Verification-Loop Sketch / Prototype.
    This module provides an architectural verification-loop sketch for
    early-exit speculation. The classifier head uses uncalibrated projection
    weights designed for testing routing and confidence exit dynamics.
"""

from typing import Tuple, List, Optional
import numpy as np


def compute_prediction_confidence(logits: np.ndarray) -> Tuple[float, float, int]:
    """Compute Shannon entropy, top-1/top-2 margin, and argmax token.

    Args:
        logits: Unnormalized vocabulary logits of shape (vocab_size,).

    Returns:
        Tuple of (entropy, margin, top_token).
    """
    m = np.max(logits)
    exp_l = np.exp(logits - m)
    probs = exp_l / np.sum(exp_l)

    # Shannon entropy: -sum(p * log(p))
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))

    # Top-1 vs Top-2 margin
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2])
    top_token = int(np.argmax(probs))

    return entropy, margin, top_token



class EarlyExitClassifier:
    """Evaluates intermediate layers and exits dynamically upon high confidence."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        entropy_threshold: float = 0.2,
        margin_threshold: float = 0.8,
    ):
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        limit = np.sqrt(6.0 / (hidden_dim + vocab_size))
        self.head_weight = np.random.uniform(-limit, limit, (hidden_dim, vocab_size)).astype(np.float32)

    def evaluate_exit(self, hidden_state: np.ndarray) -> Tuple[bool, int, float]:
        """Project intermediate state to vocabulary and check early exit criteria.

        Args:
            hidden_state: Intermediate layer output vector (hidden_dim,).

        Returns:
            Tuple of (should_exit, predicted_token, entropy).
        """
        logits = np.dot(hidden_state, self.head_weight)
        entropy, margin, token = compute_prediction_confidence(logits)
        should_exit = bool(entropy < self.entropy_threshold or margin > self.margin_threshold)
        return should_exit, token, entropy
