"""ALiBi (Attention with Linear Biases) positional bias dispatch helpers."""

from __future__ import annotations

import numpy as np


def get_alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute ALiBi head slopes following Press et al. (2021)."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    def _slopes(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start * (start**i) for i in range(n)]

    if (num_heads & (num_heads - 1)) == 0:
        return np.array(_slopes(num_heads), dtype=np.float32)
    n = 2 ** int(np.floor(np.log2(num_heads)))
    base = _slopes(n)
    extra = _slopes(2 * n)[0::2][: num_heads - n]
    return np.array(base + extra, dtype=np.float32)


def build_alibi_bias(
    num_heads: int,
    q_len: int,
    k_len: int | None = None,
    causal: bool = True,
    offset: int = 0,
) -> np.ndarray:
    """Build ALiBi positional bias tensor of shape (num_heads, q_len, k_len)."""
    if q_len <= 0:
        raise ValueError(f"q_len must be positive, got {q_len}")
    if k_len is None:
        k_len = q_len + offset
    if k_len <= 0 or k_len < offset:
        raise ValueError(f"Invalid k_len={k_len} for offset={offset}")
    slopes = get_alibi_slopes(num_heads)
    q_pos = (np.arange(q_len, dtype=np.float32) + offset)[:, None]
    k_pos = np.arange(k_len, dtype=np.float32)[None, :]
    rel_pos = k_pos - q_pos if causal else -np.abs(q_pos - k_pos)
    bias = (slopes[:, None, None] * rel_pos[None, :, :]).astype(np.float32)
    if causal:
        future_mask = k_pos > q_pos
        bias = np.where(future_mask[None, :, :], -1e9, bias).astype(np.float32)
    return bias


def apply_alibi_bias(
    scores: np.ndarray,
    bias: np.ndarray | None = None,
    slopes: np.ndarray | None = None,
    causal: bool = True,
    offset: int = 0,
) -> np.ndarray:
    """Add ALiBi positional bias to pre-softmax attention score tensor."""
    if scores.ndim < 3:
        raise ValueError(f"scores tensor must have at least 3 dims, got {scores.ndim}")
    num_heads = scores.shape[-3]
    q_len, k_len = scores.shape[-2], scores.shape[-1]
    if bias is None:
        if slopes is None:
            slopes = get_alibi_slopes(num_heads)
        bias = build_alibi_bias(
            num_heads, q_len, k_len, causal=causal, offset=offset
        ).astype(scores.dtype)
    return scores + bias


class AlibiPositionalBias:
    """Cached ALiBi positional bias generator for attention layers."""

    def __init__(self, num_heads: int, causal: bool = True) -> None:
        self.num_heads = int(num_heads)
        self.causal = bool(causal)
        self.slopes = get_alibi_slopes(self.num_heads)

    def build_bias(
        self, q_len: int, k_len: int | None = None, offset: int = 0
    ) -> np.ndarray:
        return build_alibi_bias(
            self.num_heads, q_len, k_len, causal=self.causal, offset=offset
        )

    def apply(self, scores: np.ndarray, offset: int = 0) -> np.ndarray:
        return apply_alibi_bias(
            scores, slopes=self.slopes, causal=self.causal, offset=offset
        )


def dispatch_alibi_bias(
    scores: np.ndarray, causal: bool = True, offset: int = 0
) -> np.ndarray:
    """Dispatched entry point to apply ALiBi bias to attention scores."""
    return apply_alibi_bias(scores, causal=causal, offset=offset)


__all__ = [
    "AlibiPositionalBias",
    "apply_alibi_bias",
    "build_alibi_bias",
    "dispatch_alibi_bias",
    "get_alibi_slopes",
]
