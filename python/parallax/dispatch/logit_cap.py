"""Logit soft-capping and tanh-capping dispatch operations."""

from __future__ import annotations

from typing import Any
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad



def _dispatch_logit_cap(x: np.ndarray, cap: float) -> np.ndarray:
    """Route logit soft capping through backend dispatch engine if available."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("logit_cap", x, cap=cap)
    except Exception:
        scaled = x / cap
        return (cap * np.tanh(scaled)).astype(x.dtype)


def logit_soft_cap(x: np.ndarray, cap: float = 30.0) -> np.ndarray:
    """Apply hyperbolic tangent soft-capping to logits: cap * tanh(x / cap)."""
    cap_val = float(cap)
    if cap_val <= 0.0:
        raise ValueError("cap must be strictly positive")
    return _dispatch_logit_cap(x, cap_val)



def logit_soft_cap_backward(
    grad_out: np.ndarray,
    x: np.ndarray,
    cap: float = 30.0,
) -> np.ndarray:
    """Compute analytical backward pass gradient for logit soft capping."""
    cap_val = float(cap)
    if cap_val <= 0.0:
        raise ValueError("cap must be strictly positive")
    t = np.tanh(x / cap_val)
    grad_x = grad_out * (1.0 - t * t)
    return grad_x.astype(x.dtype)


class DispatchedLogitCap(Module):
    """Logit soft-capping module using optimal backend dispatch."""

    def __init__(self, cap: float = 30.0) -> None:
        super().__init__()
        self.cap = float(cap)
        if self.cap <= 0.0:
            raise ValueError("cap must be strictly positive")

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        raw_x = x.data if isinstance(x, Tensor) else x
        out_data = logit_soft_cap(raw_x, cap=self.cap)
        parents = (x,) if isinstance(x, Tensor) else ()
        out = Tensor(out_data, parents, "DispatchedLogitCap")
        if not out.requires_grad:
            return out

        def _backward() -> None:
            if out.grad is None or not isinstance(x, Tensor) or not x.requires_grad:
                return
            grad_x = logit_soft_cap_backward(out.grad, raw_x, cap=self.cap)
            _accumulate_grad(x, grad_x)

        out._backward = _backward
        return out


    @classmethod
    def from_capper(cls, capper: Any) -> DispatchedLogitCap:
        cap = getattr(capper, "soft_cap", getattr(capper, "cap", 30.0))
        return cls(cap=cap)

    def extra_repr(self) -> str:
        return f"cap={self.cap}"

    def __repr__(self) -> str:
        return f"DispatchedLogitCap({self.extra_repr()})"


LogitSoftCap = DispatchedLogitCap

__all__ = [
    "DispatchedLogitCap",
    "LogitSoftCap",
    "logit_soft_cap",
    "logit_soft_cap_backward",
]





