"""Top-A (threshold = alpha * p_max^2) dynamic truncation sampler."""

from typing import Tuple
import numpy as np


class TopASampler:
    """Dynamically truncates vocabulary tail when confidence is high."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)

    def filter_logits(self, logits: np.ndarray) -> np.ndarray:
        """Filter logits keeping only tokens satisfying p_i >= alpha * p_max^2."""
        exp_l = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        p_max = np.max(probs, axis=-1, keepdims=True)
        threshold = self.alpha * (p_max ** 2)
        mask = probs >= threshold
        return np.where(mask, logits, -1e9)
