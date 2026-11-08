"""DeepSeek Multi-Head Latent Attention (MLA) implementation."""

from typing import Optional, Tuple
import numpy as np


class MultiHeadLatentAttention:
    """Joint low-rank KV compression attention module (DeepSeek MLA)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        rope_dim: int = 16,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.head_dim = embed_dim // num_heads

        scale = 1.0 / np.sqrt(self.embed_dim)
        self.w_dkv = np.random.randn(self.embed_dim, self.latent_dim) * scale
        self.w_uk = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale
        self.w_uv = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale
        self.w_q = np.random.randn(self.embed_dim, self.num_heads * self.head_dim) * scale
        self.w_o = np.random.randn(self.num_heads * self.head_dim, self.embed_dim) * scale

    def compress_kv(self, x: np.ndarray) -> np.ndarray:
        """Compress input x into latent KV representation c_t^KV."""
        return np.matmul(x, self.w_dkv)

    def decompress_kv(self, c_kv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project latent vector back to distributed key and value matrices."""
        k = np.matmul(c_kv, self.w_uk)
        v = np.matmul(c_kv, self.w_uv)
        return k, v
