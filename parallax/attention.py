"""
Causal Multi-Head Self-Attention and Rotary Positional Embeddings (RoPE).

Implements the core Transformer attention mechanism with scaled dot-product,
causal upper-triangular masking, multi-head projection, and RoPE dynamics.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, unbroadcast_grad


def apply_rope(
    x: np.ndarray,
    seq_dim: int = 1,
    offset: int = 0,
    conjugate: bool = False,
    theta: float = 10000.0,
    interleaved: bool = True,
) -> np.ndarray:
    """Applies Rotary Positional Embedding (RoPE) to an ndarray.

    Supports both interleaved [0::2, 1::2] and non-interleaved [:half, half:] modes.
    """
    orig_shape = x.shape
    seq_len = orig_shape[seq_dim]
    dim = orig_shape[-1]
    assert dim % 2 == 0, "RoPE requires even embedding dimension"

    # Compute inverse frequencies
    half_dim = dim // 2
    freq_seq = np.arange(half_dim, dtype=np.float64)
    inv_freq = 1.0 / (float(theta) ** (freq_seq / half_dim))

    # Positions: [seq_len, 1] offset by starting position
    positions = (np.arange(seq_len, dtype=np.float64) + float(offset))[:, None]
    # Angles: [seq_len, half_dim]
    angles = positions * inv_freq[None, :]
    if conjugate:
        angles = -angles

    sin = np.sin(angles)
    cos = np.cos(angles)

    # Reshape sin and cos to match x broadcasting
    if x.ndim == 4:
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
    elif x.ndim == 3:
        sin = sin[None, :, :]
        cos = cos[None, :, :]

    # Split into pairs
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
    else:
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]

    # Rotate pairs
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos

    out = np.empty_like(x)
    if interleaved:
        out[..., 0::2] = rx1
        out[..., 1::2] = rx2
    else:
        out[..., :half_dim] = rx1
        out[..., half_dim:] = rx2
    return out


class CausalSelfAttention(Module):
    """Multi-Head and Grouped-Query Causal Self-Attention (GQA/MHA) with optional RoPE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads for GQA"
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_rope = use_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q projection: [embed_dim, embed_dim]
        # K, V projections: [embed_dim, kv_dim]
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj = Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[np.ndarray, np.ndarray]]]:
        # x shape: [batch_size, seq_len, embed_dim]
        B, T, C = x.shape

        # 1. Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape to multi-head:
        # Q: [B, num_heads, T, head_dim]
        # K, V: [B, num_kv_heads, T, head_dim]
        q_heads = np.transpose(q.data.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))
        k_heads = np.transpose(k.data.reshape(B, T, self.num_kv_heads, self.head_dim), (0, 2, 1, 3))
        v_heads = np.transpose(v.data.reshape(B, T, self.num_kv_heads, self.head_dim), (0, 2, 1, 3))

        past_len = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            past_len = past_k.shape[2]  # [B, num_kv_heads, past_len, head_dim]

        if self.use_rope:
            # Apply RoPE on sequence dimension with past_len offset
            q_rope_in = np.transpose(q_heads, (0, 2, 1, 3))
            k_rope_in = np.transpose(k_heads, (0, 2, 1, 3))
            q_heads = np.transpose(apply_rope(q_rope_in, seq_dim=1, offset=past_len), (0, 2, 1, 3))
            k_heads = np.transpose(apply_rope(k_rope_in, seq_dim=1, offset=past_len), (0, 2, 1, 3))

        # Update and concatenate KV cache if provided
        updated_kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k_heads = np.concatenate([past_k, k_heads], axis=2)
            v_heads = np.concatenate([past_v, v_heads], axis=2)
            updated_kv_cache = (k_heads, v_heads)

        total_kv_len = k_heads.shape[2]

        # For GQA, expand/repeat KV heads to match num_heads
        if self.num_queries_per_kv > 1:
            # Repeat along head axis: [B, num_kv_heads, total_kv_len, head_dim] -> [B, num_heads, total_kv_len, head_dim]
            k_heads_expanded = np.repeat(k_heads, self.num_queries_per_kv, axis=1)
            v_heads_expanded = np.repeat(v_heads, self.num_queries_per_kv, axis=1)
        else:
            k_heads_expanded = k_heads
            v_heads_expanded = v_heads

        # 3. Scaled dot-product: Q @ K^T -> [B, num_heads, T, total_kv_len]
        scores = (q_heads @ np.swapaxes(k_heads_expanded, -1, -2)) * self.scale

        # 4. Causal masking
        if T > 1 or past_len > 0:
            q_indices = np.arange(past_len, past_len + T)[:, None]
            k_indices = np.arange(total_kv_len)[None, :]
            causal_mask = k_indices > q_indices  # [T, total_kv_len]
            scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

        # 5. Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.last_attn_weights = attn_weights

        # 6. Weighted values: Attn @ V -> [B, num_heads, T, head_dim]
        out_heads = attn_weights @ v_heads_expanded

        # 7. Merge heads: [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, embed_dim]
        merged = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, C)

        attn_out = Tensor(merged, (q, k, v), "CausalSelfAttention")

        def _backward() -> None:
            # Analytical backpropagation through attention mechanism
            if kv_cache is not None:
                # KV cached incremental step during inference is strictly forward/eval
                return

            dout = attn_out.grad  # [B, T, C]
            if dout is None:
                return

            # Reshape dout into heads: [B, num_heads, T, head_dim]
            dout_heads = np.transpose(dout.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))

            # dL/dV_expanded = Attn^T @ dL/dOut
            dv_expanded = np.swapaxes(attn_weights, -1, -2) @ dout_heads

            # Sum gradients back across repeated KV heads for GQA
            if self.num_queries_per_kv > 1:
                # dv_expanded: [B, num_kv_heads * group_size, total_kv_len, head_dim]
                B_dim, _, T_dim, H_dim = dv_expanded.shape
                dv_heads = dv_expanded.reshape(B_dim, self.num_kv_heads, self.num_queries_per_kv, T_dim, H_dim).sum(axis=2)
            else:
                dv_heads = dv_expanded

            dv = np.transpose(dv_heads, (0, 2, 1, 3)).reshape(B, T, self.kv_dim)

            # dL/dAttn = dL/dOut @ V_expanded^T
            d_attn = dout_heads @ np.swapaxes(v_heads_expanded, -1, -2)

            # Backprop through softmax:
            sum_dattn_attn = np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
            d_scores = attn_weights * (d_attn - sum_dattn_attn)
            if T > 1 or past_len > 0:
                d_scores = np.where(causal_mask[None, None, :, :], 0.0, d_scores)
            d_scores = d_scores * self.scale

            # dL/dQ = d_scores @ K_expanded
            dq_heads = d_scores @ k_heads_expanded
            if self.use_rope:
                dq_rope_in = np.transpose(dq_heads, (0, 2, 1, 3))
                dq_heads = np.transpose(apply_rope(dq_rope_in, seq_dim=1, offset=past_len, conjugate=True), (0, 2, 1, 3))
            dq = np.transpose(dq_heads, (0, 2, 1, 3)).reshape(B, T, C)

            # dL/dK_expanded = d_scores^T @ Q
            dk_expanded = np.swapaxes(d_scores, -1, -2) @ q_heads
            if self.use_rope:
                dk_rope_in = np.transpose(dk_expanded, (0, 2, 1, 3))
                dk_expanded = np.transpose(apply_rope(dk_rope_in, seq_dim=1, offset=past_len, conjugate=True), (0, 2, 1, 3))
            if self.num_queries_per_kv > 1:
                B_dim, _, T_dim, H_dim = dk_expanded.shape
                dk_heads = dk_expanded.reshape(B_dim, self.num_kv_heads, self.num_queries_per_kv, T_dim, H_dim).sum(axis=2)
            else:
                dk_heads = dk_expanded

            dk = np.transpose(dk_heads, (0, 2, 1, 3)).reshape(B, T, self.kv_dim)

            if q.requires_grad and q.grad is not None:
                q.grad += unbroadcast_grad(dq, q.shape)
            if k.requires_grad and k.grad is not None:
                k.grad += unbroadcast_grad(dk, k.shape)
            if v.requires_grad and v.grad is not None:
                v.grad += unbroadcast_grad(dv, v.shape)

        attn_out._backward = _backward

        # 8. Final projection layer
        proj_out = self.out_proj(attn_out)
        if kv_cache is not None:
            return proj_out, updated_kv_cache
        return proj_out
