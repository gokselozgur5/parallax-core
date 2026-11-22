"""Fused SwiGLU backward autograd kernel for gated MLP gradients."""

from __future__ import annotations

from typing import Any, Optional, Tuple
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


def _resolve_proj(
    proj: Any,
) -> Tuple[Optional[Tensor], Optional[Tensor], np.ndarray, Optional[np.ndarray]]:
    """Unpack Linear module, Parameter/Tensor, or ndarray projection weights."""
    if hasattr(proj, "weight"):
        w_t, b_t = proj.weight, getattr(proj, "bias", None)
        rw = w_t.data if isinstance(w_t, Tensor) else np.asarray(w_t)
        rb = b_t.data if isinstance(b_t, Tensor) else (None if b_t is None else np.asarray(b_t))
        return (
            w_t if isinstance(w_t, Tensor) else None,
            b_t if isinstance(b_t, Tensor) else None,
            rw,
            rb,
        )
    if isinstance(proj, Tensor):
        return proj, None, proj.data, None
    return None, None, np.asarray(proj), None


def gated_mlp_backward(
    dout: Any,
    x: Any,
    gate: Any,
    up: Any,
    w_gate: Any,
    w_up: Any,
    w_down: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute and accumulate backward gradients for gated MLP projections."""
    x_t, g_t, u_t = x, gate, up
    dout = dout.data if isinstance(dout, Tensor) else np.asarray(dout)
    x = x.data if isinstance(x, Tensor) else np.asarray(x)
    gate = gate.data if isinstance(gate, Tensor) else np.asarray(gate)
    up = up.data if isinstance(up, Tensor) else np.asarray(up)
    wg_t, bg_t, w_gate, _ = _resolve_proj(w_gate)
    wu_t, bu_t, w_up, _ = _resolve_proj(w_up)
    wd_t, bd_t, w_down, _ = _resolve_proj(w_down)
    dh = dout @ w_down
    dgate, dup = swiglu_backward(dh, gate, up)
    dx = (dgate @ w_gate + dup @ w_up).astype(x.dtype, copy=False)
    x_2d = x.reshape(-1, x.shape[-1])
    sig = 1.0 / (1.0 + np.exp(-np.clip(gate, -50.0, 50.0)))
    h_2d = (gate * sig * up).reshape(-1, gate.shape[-1])
    dout_2d = dout.reshape(-1, dout.shape[-1])
    dw_gate = (dgate.reshape(-1, dgate.shape[-1]).T @ x_2d).astype(w_gate.dtype)
    dw_up = (dup.reshape(-1, dup.shape[-1]).T @ x_2d).astype(w_up.dtype)
    dw_down = (dout_2d.T @ h_2d).astype(w_down.dtype)
    targets = (
        (x_t, dx), (g_t, dgate), (u_t, dup),
        (wg_t, dw_gate), (bg_t, dgate),
        (wu_t, dw_up), (bu_t, dup),
        (wd_t, dw_down), (bd_t, dout),
    )
    for t, g in targets:
        if isinstance(t, Tensor) and t.requires_grad:
            _accumulate_grad(t, g)
    return dx, dw_gate, dw_up, dw_down


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
