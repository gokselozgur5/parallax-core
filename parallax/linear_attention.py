"""
Linear Causal Attention with Recurrent Hidden State Formulation.

Implements O(N) linear attention using the kernel feature map:
phi(x) = elu(x) + 1.0 (Katharopoulos et al., 2020).
Provides both parallel causal prefix-sum training forward and recurrent
step forward for constant O(1) memory autoregressive decoding.
"""

from typing import Optional, Tuple, Union
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor


def elu_feature_map(x: np.ndarray) -> np.ndarray:
    """Non-negative kernel feature map phi(x) = elu(x) + 1."""
    return np.where(x > 0, x + 1.0, np.exp(x))


class LinearAttention(Module):
    """Linear Causal Self-Attention with dual parallel/recurrent forward paths."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        self.q_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
