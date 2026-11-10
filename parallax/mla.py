"""DeepSeek Multi-Head Latent Attention (MLA) implementation."""

from typing import Optional, Tuple
import numpy as np
from parallax.attention import apply_rope


class MultiHeadLatentAttention:
    """DeepSeek Multi-Head Latent Attention (MLA) with decoupled RoPE and latent KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        rope_dim: int = 16,
        theta: float = 10000.0,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.theta = float(theta)
        self.head_dim = embed_dim // num_heads

        scale = 1.0 / np.sqrt(self.embed_dim)
        # Latent KV down-projection and up-projections
        self.w_dkv = np.random.randn(self.embed_dim, self.latent_dim) * scale
        self.w_uk = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale
        self.w_uv = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale

        # Decoupled RoPE key projection
        self.w_kr = np.random.randn(self.embed_dim, self.rope_dim) * scale

        # Query projection (content heads + decoupled rope head)
        self.w_qc = np.random.randn(self.embed_dim, self.num_heads * self.head_dim) * scale
        self.w_qr = np.random.randn(self.embed_dim, self.num_heads * self.rope_dim) * scale

        # Output projection
        self.w_o = np.random.randn(self.num_heads * self.head_dim, self.embed_dim) * scale

    @property
    def w_q(self) -> np.ndarray:
        """Backward compatibility alias for content query weights."""
        return self.w_qc

    @w_q.setter
    def w_q(self, val: np.ndarray) -> None:
        self.w_qc = val

    def compress_kv(self, x: np.ndarray) -> np.ndarray:
        """Compress input x into latent KV representation c_t^KV."""
        return np.matmul(x, self.w_dkv)

    def decompress_kv(self, c_kv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project latent vector back to distributed key and value matrices."""
        k = np.matmul(c_kv, self.w_uk)
        v = np.matmul(c_kv, self.w_uv)
        return k, v

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
        past_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_cache: bool = False,
    ):
        """Compute multi-head latent attention output with decoupled RoPE and latent KV cache."""
        was_2d = (x.ndim == 2)
        if was_2d:
            x = x[None, :, :]
        B, T, _ = x.shape
        past_len = past_kv[0].shape[1] if past_kv is not None else 0

        # Latent KV compression and decoupled key RoPE
        c_kv_step = self.compress_kv(x)
        k_r_step = np.matmul(x, self.w_kr)
        k_r_rot_step = apply_rope(
            k_r_step, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        )

        if past_kv is not None:
            c_kv = np.concatenate([past_kv[0], c_kv_step], axis=1)
            k_r_rot = np.concatenate([past_kv[1], k_r_rot_step], axis=1)
        else:
            c_kv = c_kv_step
            k_r_rot = k_r_rot_step

        new_kv = (c_kv, k_r_rot) if use_cache else None

        # Content and decoupled rotary query heads
        q_c = np.matmul(x, self.w_qc).reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        q_r = np.matmul(x, self.w_qr).reshape(B, T, self.num_heads, self.rope_dim)
        q_r_rot = apply_rope(
            q_r, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        ).swapaxes(1, 2)

        # Decompress cached latent KV into content heads
        S = c_kv.shape[1]
        k_c, v_c = self.decompress_kv(c_kv)
        k_c = k_c.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        v_c = v_c.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        k_r_rot_heads = np.expand_dims(k_r_rot, axis=1)

        # Attention scores combining content and decoupled RoPE keys
        scores_c = np.matmul(q_c, k_c.swapaxes(-1, -2))
        scores_r = np.matmul(q_r_rot, k_r_rot_heads.swapaxes(-1, -2))
        scale = 1.0 / np.sqrt(self.head_dim + self.rope_dim)
        scores = (scores_c + scores_r) * scale

        if causal:
            q_pos = np.arange(T)[:, None] + past_len
            k_pos = np.arange(S)[None, :]
            causal_mask = k_pos <= q_pos
            scores = np.where(causal_mask[None, None, :, :], scores, -1e9)

        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        context = np.matmul(weights, v_c).swapaxes(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = np.matmul(context, self.w_o)

        if was_2d:
            out = out.squeeze(0)
        if use_cache:
            return out, new_kv
        return out
