"""Rotary Positional Embedding (RoPE) backward functional kernel with dispatch support."""

from __future__ import annotations

from typing import Optional
import numpy as np

from parallax.tensor import Tensor, _accumulate_grad
from python.parallax.dispatch.rope import apply_rotary_emb, precompute_freqs


def stabilize_theta(theta: float, min_theta: float = 1e-4) -> float:
    """Stabilize base theta ensuring strictly positive non-zero value."""
    return max(float(theta), min_theta)
