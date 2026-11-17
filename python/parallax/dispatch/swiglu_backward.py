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


def _dispatch_swiglu_backward(
    grad: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Route SwiGLU backward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch

        return dispatch("swiglu_backward", grad, gate, up)
    except Exception:
        return swiglu_backward_reference(grad, gate, up)


def swiglu_backward(
    grad: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fused SwiGLU backward gradients for gate and up tensors."""
    return _dispatch_swiglu_backward(grad, gate, up)


def gated_mlp_backward(
    dout: np.ndarray,
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute backward gradients for gated MLP projections."""
    dh = dout @ w_down
    dgate, dup = swiglu_backward(dh, gate, up)
    dx = dgate @ w_gate + dup @ w_up
    x_2d = x.reshape(-1, x.shape[-1])
    sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
    h_2d = (gate * sig * up).reshape(-1, gate.shape[-1])
    dout_2d = dout.reshape(-1, dout.shape[-1])
    dw_gate = (dgate.reshape(-1, dgate.shape[-1]).T @ x_2d).astype(w_gate.dtype)
    dw_up = (dup.reshape(-1, dup.shape[-1]).T @ x_2d).astype(w_up.dtype)
    dw_down = (dout_2d.T @ h_2d).astype(w_down.dtype)
    return dx.astype(x.dtype), dw_gate, dw_up, dw_down
