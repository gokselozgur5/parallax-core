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
