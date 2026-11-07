"""
Causal Multi-Head Self-Attention and Rotary Positional Embeddings (RoPE).

Implements the core Transformer attention mechanism with scaled dot-product,
causal upper-triangular masking, multi-head projection, and RoPE dynamics.
"""

import math
from typing import Optional, Tuple
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, unbroadcast_grad


def apply_rope(x: np.ndarray, seq_dim: int = 1) -> np.ndarray:
    """Applies Rotary Positional Embedding (RoPE) to an ndarray.

    Rotates adjacent pairs of features by position-dependent angles:
    theta_i = 10000^(-2(i-1)/d).

    Args:
        x: Array of shape [batch, seq_len, num_heads, head_dim]
           or [batch, seq_len, dim]
    """
    orig_shape = x.shape
    seq_len = orig_shape[seq_dim]
    dim = orig_shape[-1]
    assert dim % 2 == 0, "RoPE requires even embedding dimension"

    # Compute inverse frequencies
    half_dim = dim // 2
    freq_seq = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (10000.0 ** (freq_seq / half_dim))

    # Positions: [seq_len, 1]
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    # Angles: [seq_len, half_dim]
    angles = positions * inv_freq[None, :]

    sin = np.sin(angles)
    cos = np.cos(angles)

    # Reshape sin and cos to match x broadcasting
    # If x is 4D: [batch, seq_len, num_heads, dim] -> sin is [1, seq_len, 1, half_dim]
    if x.ndim == 4:
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
    elif x.ndim == 3:
        sin = sin[None, :, :]
        cos = cos[None, :, :]

    # Split into even and odd parts
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Rotate 2D pairs: [-x2, x1]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    out = np.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


class CausalSelfAttention(Module):
    """Multi-Head Causal Self-Attention with optional RoPE."""

    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projections in a single fused linear or distinct
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, embed_dim]
        B, T, C = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape to multi-head: [B, T, num_heads, head_dim] -> transpose to [B, num_heads, T, head_dim]
        q_heads = np.transpose(q.data.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))
        k_heads = np.transpose(k.data.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))
        v_heads = np.transpose(v.data.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))

        if self.use_rope:
            # Apply RoPE on sequence dimension (axis 2)
            # Temporarily transpose to [B, T, num_heads, head_dim] for rope
            q_rope_in = np.transpose(q_heads, (0, 2, 1, 3))
            k_rope_in = np.transpose(k_heads, (0, 2, 1, 3))
            q_heads = np.transpose(apply_rope(q_rope_in, seq_dim=1), (0, 2, 1, 3))
            k_heads = np.transpose(apply_rope(k_rope_in, seq_dim=1), (0, 2, 1, 3))

        # 3. Scaled dot-product: Q @ K^T -> [B, num_heads, T, T]
        scores = (q_heads @ np.swapaxes(k_heads, -1, -2)) * self.scale

        # 4. Causal masking (upper triangular mask with -inf)
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        # 5. Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.last_attn_weights = attn_weights

        # 6. Weighted values: Attn @ V -> [B, num_heads, T, head_dim]
        out_heads = attn_weights @ v_heads

        # 7. Merge heads: [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, embed_dim]
        merged = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, C)

        attn_out = Tensor(merged, (q, k, v), "CausalSelfAttention")

        def _backward() -> None:
            # Analytical backpropagation through attention mechanism
            dout = attn_out.grad  # [B, T, C]
            if dout is None:
                return

            # Reshape dout into heads: [B, num_heads, T, head_dim]
            dout_heads = np.transpose(dout.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))

            # dL/dV = Attn^T @ dL/dOut
            dv_heads = np.swapaxes(attn_weights, -1, -2) @ dout_heads
            dv = np.transpose(dv_heads, (0, 2, 1, 3)).reshape(B, T, C)

            # dL/dAttn = dL/dOut @ V^T
            d_attn = dout_heads @ np.swapaxes(v_heads, -1, -2)

            # Backprop through softmax:
            # d_scores = attn * (d_attn - sum(d_attn * attn, axis=-1, keepdims=True))
            sum_dattn_attn = np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
            d_scores = attn_weights * (d_attn - sum_dattn_attn)
            # Masked positions receive zero gradient
            d_scores = np.where(causal_mask[None, None, :, :], 0.0, d_scores)
            d_scores = d_scores * self.scale

            # dL/dQ = d_scores @ K
            dq_heads = d_scores @ k_heads
            dq = np.transpose(dq_heads, (0, 2, 1, 3)).reshape(B, T, C)

            # dL/dK = d_scores^T @ Q
            dk_heads = np.swapaxes(d_scores, -1, -2) @ q_heads
            dk = np.transpose(dk_heads, (0, 2, 1, 3)).reshape(B, T, C)

            if q.requires_grad and q.grad is not None:
                q.grad += unbroadcast_grad(dq, q.shape)
            if k.requires_grad and k.grad is not None:
                k.grad += unbroadcast_grad(dk, k.shape)
            if v.requires_grad and v.grad is not None:
                v.grad += unbroadcast_grad(dv, v.shape)

        attn_out._backward = _backward

        # 8. Final projection layer
        proj_out = self.out_proj(attn_out)
        return proj_out
