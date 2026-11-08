"""Min-P sampling module for dynamic vocabulary truncation.

Filters tokens whose probability is less than min_p * max_p, scaling dynamically
with model confidence and avoiding Top-P flat tail cutoffs.
"""

from typing import Optional
import numpy as np


def min_p_filter(
    logits: np.ndarray,
    min_p: float = 0.05,
    temperature: float = 1.0,
) -> np.ndarray:
    """Filter logits using Min-P dynamic relative probability threshold.

    Args:
        logits: Unnormalized logit array of shape (vocab_size,).
        min_p: Threshold relative to maximum token probability (0.0 to 1.0).
        temperature: Softmax sampling temperature.

    Returns:
        Renormalized probability distribution of shape (vocab_size,).
    """
    if temperature <= 0.0:
        # Greedy argmax
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[np.argmax(logits)] = 1.0
        return probs

    scaled_logits = logits / temperature
    m = np.max(scaled_logits)
    exp_logits = np.exp(scaled_logits - m)
    probs = exp_logits / np.sum(exp_logits)

    p_max = np.max(probs)
    threshold = min_p * p_max

    # Zero out probabilities below threshold
    mask = probs >= threshold
    filtered_probs = np.where(mask, probs, 0.0)

    total_prob = np.sum(filtered_probs)
    if total_prob == 0.0:
        filtered_probs[np.argmax(logits)] = 1.0
        return filtered_probs

    return filtered_probs / total_prob
