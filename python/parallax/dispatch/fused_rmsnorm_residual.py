"""Fused RMSNorm forward and backward residual kernel with dispatch support."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)


def fused_rmsnorm_residual_forward_reference(
    x: np.ndarray,
    residual: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical fused residual addition and RMSNorm forward pass."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual
    variance = np.mean(fused * fused, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(variance + eps_val)
    normed = (fused * inv_rms) * weight
    return normed.astype(fused.dtype), fused
