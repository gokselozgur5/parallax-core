"""Rotary Positional Embedding (RoPE) backward functional kernel with dispatch support."""

from __future__ import annotations

from typing import Optional
import numpy as np

from parallax.tensor import Tensor, _accumulate_grad
from python.parallax.dispatch.rope import apply_rotary_emb, precompute_freqs


def stabilize_theta(theta: float, min_theta: float = 1e-4) -> float:
    """Stabilize base theta ensuring strictly positive non-zero value."""
    return max(float(theta), min_theta)


def rope_backward_reference(
    dout: np.ndarray,
    cos: Optional[np.ndarray] = None,
    sin: Optional[np.ndarray] = None,
    offset: int = 0,
    interleaved: bool = False,
    theta: float = 10000.0,
) -> np.ndarray:
    """Compute analytical input gradient for rotary positional embedding."""
    theta_val = stabilize_theta(theta)
    t, half = dout.shape[0], dout.shape[-1] // 2
    if cos is None or sin is None:
        cos, sin = precompute_freqs(dout.shape[-1], max_seq_len=offset + t, theta=theta_val)
    c = cos[offset : offset + t].reshape((t,) + (1,) * (dout.ndim - 2) + (half,))
    s = sin[offset : offset + t].reshape((t,) + (1,) * (dout.ndim - 2) + (half,))
    dx = np.empty_like(dout)
