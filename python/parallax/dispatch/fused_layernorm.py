"""Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring strictly positive bounded value."""
    return max(float(eps), min_eps)


def rms_weight_scale(weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale weight tensor by its root-mean-square norm along the last axis."""
    eps_val = stabilize_epsilon(eps)
    rms = np.sqrt(np.mean(weight * weight, axis=-1, keepdims=True) + eps_val)
    return weight / rms
