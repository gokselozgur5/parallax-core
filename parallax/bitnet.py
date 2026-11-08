"""
BitNet b1.58 (1.58-bit LLMs: Ma et al., 2024) mathematical primitives.
"""

from typing import Tuple
import numpy as np


def quantize_weights_ternary(weight: np.ndarray) -> Tuple[np.ndarray, float]:
    """Quantizes continuous weights into ternary values {-1, 0, +1} using average absolute mean scaling."""
    # gamma = mean(|W|)
    gamma = float(np.mean(np.abs(weight))) + 1e-12
    # Scaled and rounded into {-1, 0, +1}
    w_scaled = weight / gamma
    w_ternary = np.clip(np.round(w_scaled), -1.0, 1.0)
    return w_ternary, gamma
