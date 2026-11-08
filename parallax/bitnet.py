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


def quantize_activations_int8(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantizes activation vectors to 8-bit integers using per-token Absmax scaling."""
    # scale = 127 / max(|x|) per token along feature axis
    max_val = np.max(np.abs(x), axis=-1, keepdims=True) + 1e-12
    scale = 127.0 / max_val
    x_quant = np.clip(np.round(x * scale), -128.0, 127.0)
    return x_quant, scale

