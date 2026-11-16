"""Logit soft-capping and tanh-capping dispatch operations."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


def logit_soft_cap(x: np.ndarray, cap: float = 30.0) -> np.ndarray:
    """Apply hyperbolic tangent soft-capping to logits: cap * tanh(x / cap)."""
    cap_val = float(cap)
    scaled = x / cap_val
    return (cap_val * np.tanh(scaled)).astype(x.dtype)
