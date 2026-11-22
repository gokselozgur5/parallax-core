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
    dtype = np.result_type(gate.dtype, up.dtype, grad.dtype, np.float32)
    g_arr = np.asarray(gate, dtype=dtype)
    u_arr = np.asarray(up, dtype=dtype)
    dout_arr = np.asarray(grad, dtype=dtype)
    sig = 1.0 / (1.0 + np.exp(-np.clip(g_arr, -50.0, 50.0)))
    silu = g_arr * sig
    dsilu = sig * (1.0 + g_arr * (1.0 - sig))
    dgate = (dout_arr * u_arr * dsilu).astype(dtype, copy=False)
    dup = (dout_arr * silu).astype(dtype, copy=False)
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


def dispatched_swiglu_backward(
    grad_output: Tensor | np.ndarray,
    gate: Tensor,
    up: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Propagate backward gradients to gate and up Tensors."""
    g_arr = grad_output.data if isinstance(grad_output, Tensor) else grad_output
    dgate, dup = swiglu_backward(g_arr, gate.data, up.data)
    if gate.requires_grad:
        _accumulate_grad(gate, dgate)
    if up.requires_grad:
        _accumulate_grad(up, dup)
    return Tensor(dgate), Tensor(dup)


class FusedSwiGLUBackward:
    """Dispatched backward helper and gradient accumulator for SwiGLU."""

    backward = staticmethod(swiglu_backward)
    mlp_backward = staticmethod(gated_mlp_backward)
    reference = staticmethod(swiglu_backward_reference)


__all__ = [
    "FusedSwiGLUBackward",
    "dispatched_swiglu_backward",
    "gated_mlp_backward",
    "swiglu_backward",
    "swiglu_backward_reference",
]
