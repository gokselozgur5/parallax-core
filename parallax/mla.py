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

    def forward(self, x: np.ndarray, causal: bool = True) -> np.ndarray:
        """Compute multi-head latent attention output."""
        seq_len, _ = x.shape
        q = np.matmul(x, self.w_q)
        c_kv = self.compress_kv(x)
        k, v = self.decompress_kv(c_kv)

        # Reshape to (num_heads, seq_len, head_dim)
        qh = q.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)
        kh = k.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)
        vh = v.reshape(seq_len, self.num_heads, self.head_dim).swapaxes(0, 1)

        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.matmul(qh, kh.swapaxes(-1, -2)) * scale
        if causal:
            mask = np.tril(np.ones((seq_len, seq_len)))
            scores = np.where(mask == 1, scores, -1e9)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        context = np.matmul(weights, vh).swapaxes(0, 1).reshape(seq_len, -1)
        return np.matmul(context, self.w_o)
