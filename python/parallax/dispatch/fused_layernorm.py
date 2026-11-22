"""Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring strictly positive bounded value."""
    return max(float(eps), min_eps)


def rms_weight_scale(weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale weight tensor by its root-mean-square norm along the last axis."""
    eps_val = stabilize_epsilon(eps)
    rms = np.sqrt(np.mean(weight * weight, axis=-1, keepdims=True) + eps_val)
    return weight / rms


def rms_weight_scale_backward(
    dout: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute gradient with respect to original unscaled weight."""
    eps_val = stabilize_epsilon(eps)
    mean_sq = np.mean(weight * weight, axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + eps_val)
    w_hat = weight / rms
    dot = np.mean(dout * w_hat, axis=-1, keepdims=True)
    return (dout - w_hat * dot) / rms


def _dispatch_fused_layernorm(
    x: np.ndarray,
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float,
) -> np.ndarray:
    """Route fused layernorm to backend dispatcher or fallback to numpy."""
    try:
        from parallax.backends.dispatch import dispatch
        return dispatch("layernorm", x, weight=weight, bias=bias, eps=eps)
    except Exception:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        normed = (x - mean) * inv_std
        if weight is not None:
            normed = normed * weight
        if bias is not None:
            normed = normed + bias
        return normed


def fused_layernorm_forward(
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    rms_scale: bool = False,
    return_residual: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute LayerNorm forward pass with residual addition and bias fusion."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    orig_dtype = fused.dtype
    target_dtype = np.float32 if not np.issubdtype(orig_dtype, np.floating) else orig_dtype
    acc = fused.astype(np.float32) if orig_dtype == np.float16 or not np.issubdtype(orig_dtype, np.floating) else fused
    w_eff = rms_weight_scale(weight, eps=eps_val) if weight is not None and rms_scale else weight
    w_acc = w_eff.astype(acc.dtype) if w_eff is not None else None
    b_acc = bias.astype(acc.dtype) if bias is not None else None
    normed = _dispatch_fused_layernorm(acc, w_acc, b_acc, eps_val)
    out = normed.astype(target_dtype)
    if return_residual:
        return out, fused
    return out


def _layernorm_dx(
    dout: np.ndarray,
    fused: np.ndarray,
    w_eff: Optional[np.ndarray],
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute LayerNorm input gradient and normalized activations."""
    mean = np.mean(fused, axis=-1, keepdims=True)
    var = np.var(fused, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (fused - mean) * inv_std
    d_xhat = dout * w_eff if w_eff is not None else dout
    mean_d = np.mean(d_xhat, axis=-1, keepdims=True)
    mean_dxhat = np.mean(d_xhat * x_hat, axis=-1, keepdims=True)
    dx = inv_std * (d_xhat - mean_d - x_hat * mean_dxhat)
    return dx, x_hat


def _layernorm_param_grads(
    dout: np.ndarray,
    x_hat: np.ndarray,
    weight: Optional[np.ndarray],
    bias: Optional[np.ndarray],
    eps: float,
    rms_scale: bool,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute gradients for weight and bias with optional RMS scale."""
    dw: Optional[np.ndarray] = None
    if weight is not None:
        dw_eff = np.sum(dout * x_hat, axis=tuple(range(dout.ndim - weight.ndim)))
        dw = rms_weight_scale_backward(dw_eff, weight, eps=eps) if rms_scale else dw_eff
    db: Optional[np.ndarray] = None
    if bias is not None:
        db = np.sum(dout, axis=tuple(range(dout.ndim - bias.ndim)))
    return dw, db


def fused_layernorm_backward(
    dout: np.ndarray,
    x: np.ndarray,
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    residual: Optional[np.ndarray] = None,
    eps: float = 1e-5,
    rms_scale: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute analytical backward pass for fused LayerNorm."""
    eps_val = stabilize_epsilon(eps)
    fused = x + residual if residual is not None else x
    w_eff = rms_weight_scale(weight, eps=eps_val) if weight is not None and rms_scale else weight
    dx, x_hat = _layernorm_dx(dout, fused, w_eff, eps_val)
    dw, db = _layernorm_param_grads(dout, x_hat, weight, bias, eps_val, rms_scale)
    dres = dx if residual is not None else None
    return dx, dw, db, dres


def _accumulate_norm_grads(
    x: Any,
    residual: Any,
    weight: Any,
    bias: Any,
    grads: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
) -> None:
    """Accumulate computed gradients into Tensor operands."""
    dx, dw, db, dres = grads
    if isinstance(x, Tensor) and x.requires_grad:
        _accumulate_grad(x, dx)
    if isinstance(residual, Tensor) and residual.requires_grad and dres is not None:
        _accumulate_grad(residual, dres)
    if isinstance(weight, Tensor) and weight.requires_grad and dw is not None:
        _accumulate_grad(weight, dw)
    if isinstance(bias, Tensor) and bias.requires_grad and db is not None:
        _accumulate_grad(bias, db)


def _attach_fused_layernorm_backward(
    out_t: Tensor,
    x: Any,
    weight: Any,
    bias: Any,
    residual: Any,
    eps: float,
    rms_scale: bool,
) -> None:
    """Attach autograd backward callback to output Tensor."""
    raw_x = x.data if isinstance(x, Tensor) else x
    raw_w = weight.data if isinstance(weight, Tensor) else weight
    raw_b = bias.data if isinstance(bias, Tensor) else bias
    raw_r = residual.data if isinstance(residual, Tensor) else residual

    def _backward() -> None:
        if out_t.grad is None:
            return
        grads = fused_layernorm_backward(out_t.grad, raw_x, raw_w, raw_b, residual=raw_r, eps=eps, rms_scale=rms_scale)
        _accumulate_norm_grads(x, residual, weight, bias, grads)

    out_t._backward = _backward


def _attach_residual_backward(res_t: Tensor, x: Any, residual: Any) -> None:
    """Attach pass-through autograd backward callback to fused residual Tensor."""
    if not res_t.requires_grad:
        return

    def _backward_res() -> None:
        if res_t.grad is None:
            return
        if isinstance(x, Tensor) and x.requires_grad:
            _accumulate_grad(x, res_t.grad)
        if isinstance(residual, Tensor) and residual.requires_grad:
            _accumulate_grad(residual, res_t.grad)

    res_t._backward = _backward_res


def dispatched_fused_layernorm(
    x: Tensor | np.ndarray,
    weight: Optional[Tensor | np.ndarray] = None,
    bias: Optional[Tensor | np.ndarray] = None,
    residual: Optional[Tensor | np.ndarray] = None,
    eps: float = 1e-5,
    rms_scale: bool = False,
    return_residual: bool = False,
) -> Tensor | Tuple[Tensor, Tensor] | np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Execute dispatched LayerNorm with residual, bias, and optional RMS scale."""
    raw_x = x.data if isinstance(x, Tensor) else x
    raw_w = weight.data if isinstance(weight, Tensor) else weight
    raw_b = bias.data if isinstance(bias, Tensor) else bias
    raw_r = residual.data if isinstance(residual, Tensor) else residual

    res_fwd = fused_layernorm_forward(raw_x, raw_w, raw_b, residual=raw_r, eps=eps, rms_scale=rms_scale, return_residual=return_residual)
    if not any(isinstance(p, Tensor) for p in (x, residual, weight, bias)):
        return res_fwd

    out_arr: np.ndarray = res_fwd[0] if isinstance(res_fwd, tuple) else res_fwd
    parents = tuple(p for p in (x, residual, weight, bias) if isinstance(p, Tensor))
    out_t = Tensor(out_arr, parents, "DispatchedFusedLayerNorm")
    _attach_fused_layernorm_backward(out_t, x, weight, bias, residual, eps, rms_scale)
    if return_residual and isinstance(res_fwd, tuple):
        parents_res = tuple(p for p in (x, residual) if isinstance(p, Tensor))
        res_t = Tensor(res_fwd[1], parents_res, "ResidualFused", requires_grad=False)
        _attach_residual_backward(res_t, x, residual)
        return out_t, res_t
    return out_t


class DispatchedFusedLayerNorm(Module):
    """Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        rms_scale: bool = False,
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.rms_scale = bool(rms_scale)
        self.weight: Optional[Parameter] = None
        self.bias: Optional[Parameter] = None
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32), label="FusedLN.weight")
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32), label="FusedLN.bias")

    def forward(
        self,
        x: Tensor | np.ndarray,
        residual: Optional[Tensor | np.ndarray] = None,
    ) -> Tensor:
        res = dispatched_fused_layernorm(
            x,
            weight=self.weight,
            bias=self.bias,
            residual=residual,
            eps=self.eps,
            rms_scale=self.rms_scale,
        )
        if isinstance(res, Tensor):
            return res
        if isinstance(res, np.ndarray):
            return Tensor(res)
        raise TypeError("Unexpected return type from dispatched_fused_layernorm")

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, rms_scale={self.rms_scale}"
        )

    def __repr__(self) -> str:
        return f"DispatchedFusedLayerNorm({self.extra_repr()})"

    @classmethod
    def from_layernorm(cls, norm: Any, rms_scale: bool = False) -> DispatchedFusedLayerNorm:
        shape = getattr(norm, "normalized_shape", getattr(norm, "dim", None))
        if shape is None and hasattr(norm, "weight") and norm.weight is not None:
            shape = norm.weight.shape
        shape = shape or (0,)
        eps = getattr(norm, "eps", 1e-5)
        affine = getattr(norm, "weight", None) is not None
        has_bias = getattr(norm, "bias", None) is not None
        mod = cls(shape, eps=eps, elementwise_affine=affine, bias=has_bias, rms_scale=rms_scale)
        if affine and hasattr(norm, "weight") and norm.weight is not None:
            mod.weight = Parameter(np.asarray(getattr(norm.weight, "data", norm.weight)).copy())
        if has_bias and hasattr(norm, "bias") and norm.bias is not None:
            mod.bias = Parameter(np.asarray(getattr(norm.bias, "data", norm.bias)).copy())
        return mod


FusedLayerNormRMS = DispatchedFusedLayerNorm

__all__ = [
    "DispatchedFusedLayerNorm",
    "FusedLayerNormRMS",
    "dispatched_fused_layernorm",
    "fused_layernorm_backward",
    "fused_layernorm_forward",
    "rms_weight_scale",
    "rms_weight_scale_backward",
    "stabilize_epsilon",
]
