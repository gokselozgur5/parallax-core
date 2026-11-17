"""Fused SwiGLU backward autograd kernel for gated MLP gradients."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from parallax.tensor import Tensor, _accumulate_grad


def swiglu_backward_reference(
    grad: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical backward gradients for SwiGLU inputs."""
    if np.iscomplexobj(gate) or np.iscomplexobj(up) or np.iscomplexobj(grad):
        raise TypeError("Complex dtypes are not supported")
    if gate.shape != up.shape:
        raise ValueError(f"Shape mismatch: {gate.shape} vs {up.shape}")
