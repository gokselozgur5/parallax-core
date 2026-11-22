"""SwiGLU activation layer and dispatch helpers."""

from __future__ import annotations

from typing import Any
import numpy as np

try:
    from parallax.native.swiglu import c_swiglu
except ImportError:
    c_swiglu = None



from parallax.nn import Module
from parallax.nn.linear import DispatchedLinear

from parallax.tensor import Tensor, _accumulate_grad



def swiglu_dispatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute fused SwiGLU activation: silu(a) * b."""
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError("Complex dtypes are not supported")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a_f32 = np.asarray(a, dtype=np.float32)
    b_f32 = np.asarray(b, dtype=np.float32)
    if c_swiglu is not None:
        try:
            return c_swiglu(a_f32, b_f32)
        except Exception:
            pass
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


class SwiGLUActivation(Module):
    """SwiGLU activation layer: computes silu(gate) * up."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, gate: Tensor, up: Tensor | None = None) -> Tensor:
        if up is None:
            axis = self.dim
            g, u = np.split(gate.data, 2, axis=axis)
            out = Tensor(swiglu_dispatch(g, u), (gate,), "SwiGLU")
            if not out.requires_grad:
                return out

            def _backward() -> None:
                if out.grad is None or not gate.requires_grad:
                    return
                ga, gb = swiglu_backward_dispatch(out.grad, g, u)
                _accumulate_grad(gate, np.concatenate((ga, gb), axis=axis))

            out._backward = _backward
            return out
        return dispatched_swiglu(gate, up)


class DispatchedSwiGLU(Module):
    """Dispatched SwiGLU FFN module: (silu(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * embed_dim) / 3)
            hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.hidden_dim = hidden_dim
        self.w_gate = DispatchedLinear(embed_dim, hidden_dim, bias=bias)
        self.w_up = DispatchedLinear(embed_dim, hidden_dim, bias=bias)
        self.w_down = DispatchedLinear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        fused = dispatched_swiglu(gate, up)
        return self.w_down(fused)

    @classmethod
    def from_swiglu(cls, swiglu: Any) -> DispatchedSwiGLU:
        embed_dim = swiglu.w_gate.weight.shape[1]
        hidden_dim = swiglu.w_gate.weight.shape[0]
        has_bias = getattr(swiglu.w_gate, "bias", None) is not None
        mod = cls(embed_dim, hidden_dim, bias=has_bias)
        mod.w_gate = DispatchedLinear.from_linear(swiglu.w_gate)
        mod.w_up = DispatchedLinear.from_linear(swiglu.w_up)
        mod.w_down = DispatchedLinear.from_linear(swiglu.w_down)
        return mod

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}"

    def __repr__(self) -> str:
        return f"DispatchedSwiGLU({self.extra_repr()})"


__all__ = [
    "DispatchedSwiGLU",
    "SwiGLUActivation",
    "dispatched_swiglu",
    "swiglu_backward_dispatch",
    "swiglu_dispatch",
]






