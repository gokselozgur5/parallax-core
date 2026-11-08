"""Dynamic Speculative Depth with Shannon entropy early-exit classifier.

Computes predictive confidence (entropy or top-1/top-2 margin) from intermediate
hidden layers and exits early when certainty exceeds a threshold.
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
