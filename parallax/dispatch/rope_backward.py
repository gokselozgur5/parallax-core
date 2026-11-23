"""Rotary Positional Embedding (RoPE) backward functional kernel with dispatch support."""

from __future__ import annotations

from typing import Optional
import numpy as np

from parallax.tensor import Tensor, _accumulate_grad
from parallax.dispatch.rope import apply_rotary_emb, precompute_freqs


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
    if interleaved:
        d1, d2 = dout[..., 0::2], dout[..., 1::2]
        dx[..., 0::2] = d1 * c + d2 * s
        dx[..., 1::2] = -d1 * s + d2 * c
    else:
        d1, d2 = dout[..., :half], dout[..., half:]
        dx[..., :half] = d1 * c + d2 * s
        dx[..., half:] = -d1 * s + d2 * c
    return dx.astype(dout.dtype)


def _dispatch_rope_backward(
    dout: np.ndarray,
    cos: Optional[np.ndarray],
    sin: Optional[np.ndarray],
    offset: int,
    interleaved: bool,
    theta: float,
) -> np.ndarray:
    """Route RoPE backward through backend dispatch or fallback."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch(
            "rope_backward", dout, cos=cos, sin=sin, offset=offset, interleaved=interleaved, theta=theta
        )
    except Exception:
        return rope_backward_reference(
            dout, cos=cos, sin=sin, offset=offset, interleaved=interleaved, theta=theta
        )


def rope_backward(
    dout: np.ndarray,
    cos: Optional[np.ndarray] = None,
    sin: Optional[np.ndarray] = None,
    offset: int = 0,
    interleaved: bool = False,
    theta: float = 10000.0,
) -> np.ndarray:
    """Execute functional backward pass for Rotary Positional Embedding."""
    theta_val = stabilize_theta(theta)
    return _dispatch_rope_backward(
        dout,
        cos=cos,
        sin=sin,
        offset=offset,
        interleaved=interleaved,
        theta=theta_val,
    )


def dispatched_rope(
    x: Tensor | np.ndarray,
    offset: int = 0,
    interleaved: bool = False,
    theta: float = 10000.0,
) -> Tensor:
    """Dispatched RoPE functional forward with autograd backward hook."""
    theta_val = stabilize_theta(theta)
    raw_x = x.data if isinstance(x, Tensor) else x
    t = raw_x.shape[0]
    dim = raw_x.shape[-1]
    cos, sin = precompute_freqs(dim, max_seq_len=offset + t, theta=theta_val)
    out_data = apply_rotary_emb(raw_x, cos, sin, offset=offset, interleaved=interleaved)
    parents = (x,) if isinstance(x, Tensor) else ()
    out = Tensor(out_data, parents, "dispatched_rope")
    if not out.requires_grad:
        return out

    def _backward() -> None:
        if out.grad is None:
            return
        dx = rope_backward(
            out.grad,
            cos=cos,
            sin=sin,
            offset=offset,
            interleaved=interleaved,
            theta=theta_val,
        )
        if isinstance(x, Tensor) and x.requires_grad:
            _accumulate_grad(x, dx)

    out._backward = _backward
    return out


class DispatchedRoPEBackward:
    """Helper container for executing RoPE backward operations."""

    def __init__(
        self, offset: int = 0, interleaved: bool = False, theta: float = 10000.0
    ) -> None:
        self.offset = offset
        self.interleaved = interleaved
        self.theta = theta

    def backward(
        self,
        dout: np.ndarray,
        cos: Optional[np.ndarray] = None,
        sin: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return rope_backward(
            dout,
            cos=cos,
            sin=sin,
            offset=self.offset,
            interleaved=self.interleaved,
            theta=self.theta,
        )


__all__ = [
    "DispatchedRoPEBackward",
    "dispatched_rope",
    "rope_backward",
    "rope_backward_reference",
    "stabilize_theta",
]
