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
