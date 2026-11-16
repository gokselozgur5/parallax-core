"""Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring positive strictly-bounded value."""
    eps_val = float(eps)
    return max(eps_val, min_eps)


def fused_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute RMSNorm with fused residual addition and epsilon stabilization."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    orig_dtype = fused.dtype
    acc = fused.astype(np.float32) if orig_dtype in (np.float16, np.int32) else fused
    variance = np.mean(acc * acc, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(np.maximum(variance, 0.0) + eps_val)
    normed = (fused * inv_rms) * weight
    return normed.astype(orig_dtype)

