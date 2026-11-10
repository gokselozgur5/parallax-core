"""
BitAttention: 1.58-bit Ternary Multi-Head Attention Fusion.

Fuses BitNet b1.58 ternary linear projections {-1, 0, +1} (Ma et al., 2024)
with Causal Multi-Head / Grouped-Query Attention (MHA/GQA), FlashAttention-2 online
softmax acceleration, RoPE rotary position embeddings, and KV-cache autoregressive decoding.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np

from parallax.attention import apply_rope
from parallax.bitnet import BitLinear
from parallax.flash_attn import flash_attention_backward, flash_attention_forward
from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad


class BitAttention(Module):
    """Multi-Head / Grouped-Query Causal Attention with 1.58-bit ternary projections.

    All Q, K, V and Out projections are replaced by BitLinear layers, computing
    integer addition-dominated matrix products during forward passes while supporting
    exact straight-through estimator (STE) gradient flow during backpropagation.
    FlashAttention online softmax statistics and KV caching are natively supported.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = True,
        use_flash_attn: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for GQA")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.use_rope = use_rope
        self.qkv_bias = qkv_bias
        self.rope_theta = float(rope_theta)
        self.rope_interleaved = rope_interleaved
        self.use_flash_attn = use_flash_attn
        self.sliding_window = sliding_window
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1.58-bit ternary projections for Q, K, V and Out
        self.q_proj = BitLinear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = BitLinear(embed_dim, self.kv_dim, bias=qkv_bias)
        self.v_proj = BitLinear(embed_dim, self.kv_dim, bias=qkv_bias)
        self.out_proj = BitLinear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rope_offset: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[np.ndarray, np.ndarray]]]:
        """Forward pass of BitAttention.

        Parameters
        ----------
        x : Tensor of shape [batch_size, seq_len, embed_dim]
        kv_cache : Optional tuple of (past_k, past_v) with shape [B, num_kv_heads, past_len, head_dim]
        rope_offset : Optional starting sequence offset for RoPE rotary embeddings

        Returns
        -------
        Tensor or tuple of (Tensor, updated_kv_cache)
        """
        B, T, C = x.shape

        # 1. Project inputs to Q, K, V using BitLinear ternary projections
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

        offset = past_len if rope_offset is None else rope_offset

        if self.use_rope:
            q_rope_in = np.transpose(q_heads, (0, 2, 1, 3))
            k_rope_in = np.transpose(k_heads, (0, 2, 1, 3))
            q_heads = np.transpose(
                apply_rope(
                    q_rope_in,
                    seq_dim=1,
                    offset=offset,
                    theta=self.rope_theta,
                    interleaved=self.rope_interleaved,
                ),
                (0, 2, 1, 3),
            )
            k_heads = np.transpose(
                apply_rope(
                    k_rope_in,
                    seq_dim=1,
                    offset=offset,
                    theta=self.rope_theta,
                    interleaved=self.rope_interleaved,
                ),
                (0, 2, 1, 3),
            )

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
            k_heads_expanded = np.repeat(k_heads, self.num_queries_per_kv, axis=1)
            v_heads_expanded = np.repeat(v_heads, self.num_queries_per_kv, axis=1)
        else:
            k_heads_expanded = k_heads
            v_heads_expanded = v_heads

        # 3. Scaled dot-product attention (FlashAttention or standard causal softmax)
        if self.use_flash_attn and past_len == 0:
            out_heads, lse = flash_attention_forward(
                q_heads,
                k_heads_expanded,
                v_heads_expanded,
                is_causal=True,
            )
        else:
            scores = (q_heads @ np.swapaxes(k_heads_expanded, -1, -2)) * self.scale

            if T > 1 or past_len > 0:
                q_indices = np.arange(past_len, past_len + T)[:, None]
                k_indices = np.arange(total_kv_len)[None, :]
                causal_mask = k_indices > q_indices  # [T, total_kv_len]
                if self.sliding_window is not None:
                    window_mask = (q_indices - k_indices) >= self.sliding_window
                    causal_mask = causal_mask | window_mask
                scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            self.last_attn_weights = attn_weights
            out_heads = attn_weights @ v_heads_expanded

        # 4. Merge heads: [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, embed_dim]
        merged = np.transpose(out_heads, (0, 2, 1, 3)).reshape(B, T, C)

        attn_out = Tensor(merged, (q, k, v), "BitAttention")

        def _backward() -> None:
            if kv_cache is not None:
                return

            dout = attn_out.grad
            if dout is None:
                return

            dout_heads = np.transpose(dout.reshape(B, T, self.num_heads, self.head_dim), (0, 2, 1, 3))

            if self.use_flash_attn and past_len == 0:
                dq_heads, dk_heads_expanded, dv_expanded = flash_attention_backward(
                    dout_heads,
                    out_heads,
                    q_heads,
                    k_heads_expanded,
                    v_heads_expanded,
                    lse,
                    is_causal=True,
                )
            else:
                dv_expanded = np.swapaxes(attn_weights, -1, -2) @ dout_heads

            if self.num_queries_per_kv > 1:
                B_dim, _, T_dim, H_dim = dv_expanded.shape
                dv_heads = dv_expanded.reshape(
                    B_dim, self.num_kv_heads, self.num_queries_per_kv, T_dim, H_dim
                ).sum(axis=2)
            else:
                dv_heads = dv_expanded

            dv = np.transpose(dv_heads, (0, 2, 1, 3)).reshape(B, T, self.kv_dim)

            if not (self.use_flash_attn and past_len == 0):
                d_attn = dout_heads @ np.swapaxes(v_heads_expanded, -1, -2)
                sum_dattn_attn = np.sum(d_attn * attn_weights, axis=-1, keepdims=True)
                d_scores = attn_weights * (d_attn - sum_dattn_attn)
                if T > 1 or past_len > 0:
                    d_scores = np.where(causal_mask[None, None, :, :], 0.0, d_scores)
                d_scores = d_scores * self.scale

                dq_heads = d_scores @ k_heads_expanded

            if self.use_rope:
                dq_rope_in = np.transpose(dq_heads, (0, 2, 1, 3))
                dq_heads = np.transpose(
                    apply_rope(
                        dq_rope_in,
                        seq_dim=1,
                        offset=offset,
                        conjugate=True,
                        theta=self.rope_theta,
                        interleaved=self.rope_interleaved,
                    ),
                    (0, 2, 1, 3),
                )
            dq = np.transpose(dq_heads, (0, 2, 1, 3)).reshape(B, T, C)

            if not (self.use_flash_attn and past_len == 0):
                dk_expanded = np.swapaxes(d_scores, -1, -2) @ q_heads
            else:
                dk_expanded = dk_heads_expanded

            if self.use_rope:
                dk_rope_in = np.transpose(dk_expanded, (0, 2, 1, 3))
                dk_expanded = np.transpose(
                    apply_rope(
                        dk_rope_in,
                        seq_dim=1,
                        offset=offset,
                        conjugate=True,
                        theta=self.rope_theta,
                        interleaved=self.rope_interleaved,
                    ),
                    (0, 2, 1, 3),
                )

            if self.num_queries_per_kv > 1:
                B_dim, _, T_dim, H_dim = dk_expanded.shape
                dk_heads = dk_expanded.reshape(
                    B_dim, self.num_kv_heads, self.num_queries_per_kv, T_dim, H_dim
                ).sum(axis=2)
            else:
                dk_heads = dk_expanded

            dk = np.transpose(dk_heads, (0, 2, 1, 3)).reshape(B, T, self.kv_dim)

            if q.requires_grad:
                _accumulate_grad(q, dq)
            if k.requires_grad:
                _accumulate_grad(k, dk)
            if v.requires_grad:
                _accumulate_grad(v, dv)

        attn_out._backward = _backward

        # 5. Final projection layer through BitLinear
        proj_out = self.out_proj(attn_out)
        if kv_cache is not None:
            return proj_out, updated_kv_cache
        return proj_out
