"""Fused LayerNorm dispatch with affine scaling and residual fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring strictly positive bounded value."""
    return max(float(eps), min_eps)


def _dispatch_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    """Compute layernorm directly using reference implementation."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normed = (x - mean) * inv_std
    if weight is not None:
        normed = normed * weight
    if bias is not None:
        normed = normed + bias
    return normed


def fused_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    return_residual: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute LayerNorm with optional fused residual and affine parameters."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    orig_dtype = fused.dtype
    acc = fused.astype(np.float32) if orig_dtype in (np.float16, np.int32) else fused
    w_acc = weight.astype(acc.dtype) if weight is not None else None
    b_acc = bias.astype(acc.dtype) if bias is not None else None
    normed = _dispatch_layernorm(acc, w_acc, b_acc, eps_val)
    out = normed.astype(orig_dtype)
    if return_residual:
        return out, fused
    return out


def _backward_layernorm(
    dout: np.ndarray,
    fused: np.ndarray,
    eps: float,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    x: Optional[Tensor],
    residual: Optional[Tensor],
) -> None:
    """Compute and accumulate gradients for LayerNorm operation."""
    eps_val = stabilize_epsilon(eps)
    mean = np.mean(fused, axis=-1, keepdims=True)
    var = np.var(fused, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps_val)
    x_hat = (fused - mean) * inv_std
    d = fused.shape[-1]
    gw = dout * weight.data if weight is not None else dout
    dvar = np.sum(gw * (fused - mean) * -0.5 * (inv_std ** 3), axis=-1, keepdims=True)
    dmean = np.sum(gw * -inv_std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (fused - mean), axis=-1, keepdims=True)
    dx = gw * inv_std + dvar * 2.0 * (fused - mean) / d + dmean / d
    if weight is not None and weight.requires_grad:
        _accumulate_grad(weight, np.sum(dout * x_hat, axis=tuple(range(dout.ndim - 1))))
    if bias is not None and bias.requires_grad:
        _accumulate_grad(bias, np.sum(dout, axis=tuple(range(dout.ndim - 1))))
    if isinstance(x, Tensor) and x.requires_grad:
        _accumulate_grad(x, dx)
    if isinstance(residual, Tensor) and residual.requires_grad:
        _accumulate_grad(residual, dx)


class FusedLayerNorm(Module):
    """LayerNorm layer supporting fused affine scaling and residual addition."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32), label="LayerNorm.weight")
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32), label="LayerNorm.bias")

    def forward(
        self,
        x: Tensor | np.ndarray,
        residual: Optional[Tensor | np.ndarray] = None,
    ) -> Tensor:
        raw_x = x.data if isinstance(x, Tensor) else x
        raw_res = residual.data if isinstance(residual, Tensor) else residual
        raw_w = self.weight.data if self.weight is not None else None
        raw_b = self.bias.data if self.bias is not None else None
        eps_val = stabilize_epsilon(self.eps)
        out_data = fused_layernorm(raw_x, raw_w, raw_b, residual=raw_res, eps=eps_val)
        parents = [p for p in (x, residual, self.weight, self.bias) if isinstance(p, Tensor)]
        out = Tensor(out_data, tuple(parents), "FusedLayerNorm")
        if not out.requires_grad:
            return out
        raw_fused = raw_x + raw_res if raw_res is not None else raw_x
        out._backward = lambda: _backward_layernorm(
            out.grad, raw_fused, eps_val, self.weight, self.bias,
            x if isinstance(x, Tensor) else None,
            residual if isinstance(residual, Tensor) else None,
        ) if out.grad is not None else None
        return out

    @classmethod
    def from_layernorm(cls, norm: Any) -> FusedLayerNorm:
        shape = getattr(norm, "normalized_shape", getattr(norm, "dim", None))
        if shape is None and hasattr(norm, "weight") and norm.weight is not None:
            shape = norm.weight.shape
        shape = shape or (0,)
        eps = getattr(norm, "eps", 1e-5)
        affine = getattr(norm, "weight", None) is not None
        has_bias = getattr(norm, "bias", None) is not None
        mod = cls(shape, eps=eps, elementwise_affine=affine, bias=has_bias)
        if affine and hasattr(norm, "weight") and norm.weight is not None:
            mod.weight = Parameter(np.asarray(getattr(norm.weight, "data", norm.weight)).copy())
        if has_bias and hasattr(norm, "bias") and norm.bias is not None:
            mod.bias = Parameter(np.asarray(getattr(norm.bias, "data", norm.bias)).copy())
        return mod

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"

    def __repr__(self) -> str:
        return f"FusedLayerNorm({self.extra_repr()})"


__all__ = ["FusedLayerNorm", "fused_layernorm", "stabilize_epsilon"]
