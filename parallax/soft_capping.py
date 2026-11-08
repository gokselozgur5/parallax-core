"""Logit soft-capping kernel (Gemma-2 style cap * tanh(x / cap))."""

from typing import Tuple
import numpy as np


class LogitSoftCapper:
    """Softly bounds logit dynamic range using hyperbolic tangent."""

    def __init__(self, soft_cap: float = 30.0) -> None:
        self.soft_cap = float(soft_cap)

    def forward(self, logits: np.ndarray) -> np.ndarray:
        """Apply soft capping: cap * tanh(logits / cap)."""
        scaled = logits / self.soft_cap
        return self.soft_cap * np.tanh(scaled)

    def backward(self, grad_out: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Compute backward gradient: grad_out * (1 - tanh^2(logits / cap))."""
        scaled = logits / self.soft_cap
        tanh_val = np.tanh(scaled)
        return grad_out * (1.0 - tanh_val ** 2)
