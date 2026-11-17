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
    if grad.shape != gate.shape:
        raise ValueError(f"Grad shape mismatch: {grad.shape} vs {gate.shape}")
    g_f32 = np.asarray(gate, dtype=np.float32)
    u_f32 = np.asarray(up, dtype=np.float32)
    dout_f32 = np.asarray(grad, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-np.clip(g_f32, -50.0, 50.0)))
    silu = g_f32 * sig
    dsilu = sig * (1.0 + g_f32 * (1.0 - sig))
    dgate = (dout_f32 * u_f32 * dsilu).astype(np.float32)
    dup = (dout_f32 * silu).astype(np.float32)
    return dgate, dup
