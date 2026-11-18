"""Scaled Dot-Product Attention (SDPA) dispatch operations."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


def _validate_sdpa_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: Optional[np.ndarray] = None,
    is_causal: bool = False,
) -> None:
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must be at least 2D arrays")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"query head_dim ({query.shape[-1]}) != key head_dim ({key.shape[-1]})"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            f"key seq_len ({key.shape[-2]}) != value seq_len ({value.shape[-2]})"
        )
    if is_causal and attn_mask is not None:
        raise ValueError("Cannot specify both is_causal=True and attn_mask")


def _compute_sdpa_scores(
    query: np.ndarray,
    key: np.ndarray,
    scale: Optional[float] = None,
    attn_mask: Optional[np.ndarray] = None,
    is_causal: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    d_k = query.shape[-1]
    inv_scale = float(scale) if scale is not None else 1.0 / math.sqrt(d_k)
    scores = np.matmul(query, key.swapaxes(-2, -1)) * inv_scale
    causal_mask = None
    if is_causal:
        t_q, t_k = query.shape[-2], key.shape[-2]
        causal_mask = np.arange(t_q)[:, None] >= np.arange(t_k)[None, :]
        scores = np.where(causal_mask, scores, -1e9)
    elif attn_mask is not None:
        if attn_mask.dtype == bool:
            scores = np.where(attn_mask, scores, -1e9)
        else:
            scores = scores + attn_mask
    return scores, causal_mask, inv_scale


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Compute scaled dot-product attention."""
    if dropout_p > 0.0:
        raise ValueError("dropout_p > 0.0 is not supported")
    _validate_sdpa_inputs(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
    scores, _, _ = _compute_sdpa_scores(
        query, key, scale=scale, attn_mask=attn_mask, is_causal=is_causal
    )
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out = np.matmul(attn_weights, value)
    return out.astype(query.dtype)


def _compute_sdpa_backward_scores(
    grad_out: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: Optional[np.ndarray] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    scores, causal_mask, inv_scale = _compute_sdpa_scores(
        query, key, scale=scale, attn_mask=attn_mask, is_causal=is_causal
    )
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    grad_value = np.matmul(attn_weights.swapaxes(-2, -1), grad_out)
    grad_attn = np.matmul(grad_out, value.swapaxes(-2, -1))
    sum_grad_attn = np.sum(grad_attn * attn_weights, axis=-1, keepdims=True)
    grad_scores = attn_weights * (grad_attn - sum_grad_attn)
    if is_causal and causal_mask is not None:
        grad_scores = np.where(causal_mask, grad_scores, 0.0)
    elif attn_mask is not None and attn_mask.dtype == bool:
        grad_scores = np.where(attn_mask, grad_scores, 0.0)
    return grad_scores * inv_scale, grad_value


def sdpa_backward(
    grad_out: np.ndarray,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: Optional[np.ndarray] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical backward pass gradients for SDPA."""
    _validate_sdpa_inputs(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
    grad_scores, grad_value = _compute_sdpa_backward_scores(
        grad_out, query, key, value, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )
    grad_query = np.matmul(grad_scores, key)
    grad_key = np.matmul(grad_scores.swapaxes(-2, -1), query)
    return (
        grad_query.astype(query.dtype),
        grad_key.astype(key.dtype),
        grad_value.astype(value.dtype),
    )


class DispatchedSDPA(Module):
    """Scaled Dot-Product Attention module using dispatch."""

    def __init__(
        self,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)
        self.is_causal = bool(is_causal)
        self.scale = float(scale) if scale is not None else None

    def forward(
        self,
        query: Tensor | np.ndarray,
        key: Tensor | np.ndarray,
        value: Tensor | np.ndarray,
        attn_mask: Optional[Tensor | np.ndarray] = None,
    ) -> Tensor:
        raw_q = query.data if isinstance(query, Tensor) else query
        raw_k = key.data if isinstance(key, Tensor) else key
        raw_v = value.data if isinstance(value, Tensor) else value
        raw_mask = attn_mask.data if isinstance(attn_mask, Tensor) else attn_mask

        out_data = scaled_dot_product_attention(
            raw_q,
            raw_k,
            raw_v,
            attn_mask=raw_mask,
            dropout_p=self.dropout_p,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        parents = tuple(p for p in (query, key, value) if isinstance(p, Tensor))
        out = Tensor(out_data, parents, "DispatchedSDPA")
        if not out.requires_grad:
            return out
        def _backward() -> None:
            if out.grad is None:
                return
            gq, gk, gv = sdpa_backward(
                out.grad,
                raw_q,
                raw_k,
                raw_v,
                attn_mask=raw_mask,
                is_causal=self.is_causal,
                scale=self.scale,
            )
            if isinstance(query, Tensor) and query.requires_grad:
                _accumulate_grad(query, gq)
            if isinstance(key, Tensor) and key.requires_grad:
                _accumulate_grad(key, gk)
            if isinstance(value, Tensor) and value.requires_grad:
                _accumulate_grad(value, gv)

        out._backward = _backward
        return out


    def extra_repr(self) -> str:
        return f"dropout_p={self.dropout_p}, is_causal={self.is_causal}, scale={self.scale}"

    def __repr__(self) -> str:
        return f"DispatchedSDPA({self.extra_repr()})"


SDPAModule = DispatchedSDPA


def dispatched_sdpa(
    query: Tensor | np.ndarray,
    key: Tensor | np.ndarray,
    value: Tensor | np.ndarray,
    attn_mask: Optional[Tensor | np.ndarray] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Dispatched entry point routing SDPA computation with autograd support."""
    mod = DispatchedSDPA(dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    return mod(query, key, value, attn_mask=attn_mask)


__all__ = [
    "DispatchedSDPA",
    "SDPAModule",
    "dispatched_sdpa",
    "scaled_dot_product_attention",
    "sdpa_backward",
]
