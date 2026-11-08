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
