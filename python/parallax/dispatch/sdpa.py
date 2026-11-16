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
