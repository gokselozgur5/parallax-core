"""SwiGLU activation layer and dispatch helpers."""

from __future__ import annotations

import numpy as np

from parallax.tensor import Tensor, _accumulate_grad


def swiglu_dispatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute fused SwiGLU activation: silu(a) * b."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError("Complex dtypes are not supported")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-np.clip(a_f32, -50.0, 50.0)))
    return (a_f32 * sig * b_f32).astype(np.float32)


def swiglu_backward_dispatch(
    grad: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradients for SwiGLU inputs a and b."""
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    g_f32 = np.asarray(grad, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-np.clip(a_f32, -50.0, 50.0)))
    silu = a_f32 * sig
    dsilu = sig * (1.0 + a_f32 * (1.0 - sig))
    return (g_f32 * b_f32 * dsilu).astype(np.float32), (g_f32 * silu).astype(np.float32)


def dispatched_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Compute SwiGLU activation over Tensors with autograd tracking."""
    out_data = swiglu_dispatch(gate.data, up.data)
    out = Tensor(out_data, (gate, up), "SwiGLU")

    def _backward() -> None:
        if out.grad is None:
            return
        ga, gb = swiglu_backward_dispatch(out.grad, gate.data, up.data)
        if gate.requires_grad:
            _accumulate_grad(gate, ga)
        if up.requires_grad:
            _accumulate_grad(up, gb)

    out._backward = _backward
    return out


