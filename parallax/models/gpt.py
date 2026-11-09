"""
Decoder-only Transformer Block and ParallaxLM architecture.

Implements the standard Pre-LN Transformer architecture with RMSNorm,
causal multi-head self-attention, and MLP feed-forward blocks.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from parallax.attention import CausalSelfAttention
from parallax.nn import Linear, Module, Parameter, RMSNorm
from parallax.tensor import Tensor, unbroadcast_grad


def tensor_gelu(x: Tensor) -> Tensor:
    """Applies elementwise GELU approximation to an N-D Tensor."""
    x_data = x.data
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x_data + 0.044715 * (x_data**3))
    tanh_inner = np.tanh(inner)
    cdf = 0.5 * (1.0 + tanh_inner)
    out_data = x_data * cdf

    out = Tensor(out_data, (x,), "GELU")

    def _backward() -> None:
        if x.requires_grad and x.grad is not None:
            sech2 = 1.0 - tanh_inner**2
            d_inner = c * (1.0 + 3.0 * 0.044715 * (x_data**2))
            d_cdf = 0.5 * sech2 * d_inner
            dx = (cdf + x_data * d_cdf) * out.grad
            x.grad += unbroadcast_grad(dx, x.shape)

    out._backward = _backward
    return out


class MLP(Module):
    """Feed-Forward Network with GELU activation."""

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = tensor_gelu(self.fc1(x))
        out = self.fc2(h)
        return out


def tensor_silu(x: Tensor) -> Tensor:
    """Applies elementwise SiLU (Swish) activation: x * sigmoid(x)."""
    x_data = x.data
    sig = 1.0 / (1.0 + np.exp(-np.clip(x_data, -50.0, 50.0)))
    out_data = x_data * sig
    out = Tensor(out_data, (x,), "SiLU")

    def _backward() -> None:
        if x.requires_grad and x.grad is not None:
            # d/dx (x * sig) = sig + x * sig * (1 - sig) = sig * (1 + x * (1 - sig))
            dsilu = sig * (1.0 + x_data * (1.0 - sig))
            dx = dsilu * out.grad
            x.grad += unbroadcast_grad(dx, x.shape)

    out._backward = _backward
    return out


class SwiGLU(Module):
    """SwiGLU Feed-Forward Network: out = (SiLU(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T."""

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        # Standard LLaMA hidden dimension scaling: ~ 8/3 * embed_dim rounded to multiple of 64
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * embed_dim) / 3)
            hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)

        self.w_gate = Linear(embed_dim, hidden_dim, bias=False)
        self.w_up = Linear(embed_dim, hidden_dim, bias=False)
        self.w_down = Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = tensor_silu(self.w_gate(x))
        up = self.w_up(x)
        fused = gate * up
        out = self.w_down(fused)
        return out


class TransformerBlock(Module):
    """Pre-LN Transformer Decoder Block with selectable GELU/SwiGLU and GQA support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        use_rope: bool = True,
        use_swiglu: bool = False,
        num_experts: Optional[int] = None,
        top_k: int = 2,
        intermediate_size: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.attn = CausalSelfAttention(
            embed_dim,
            num_heads,
            num_kv_heads=num_kv_heads,
            use_rope=use_rope,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            rope_interleaved=rope_interleaved,
        )
        self.mlp_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.use_swiglu = use_swiglu
        self.num_experts = num_experts

        if num_experts is not None and num_experts > 1:
            from parallax.moe import SparseMoE
            self.mlp = SparseMoE(
                embed_dim=embed_dim,
                num_experts=num_experts,
                top_k=top_k,
                use_swiglu=use_swiglu,
                hidden_dim=intermediate_size,
            )
        elif use_swiglu:
            self.mlp = SwiGLU(embed_dim, hidden_dim=intermediate_size)
        else:
            self.mlp = MLP(embed_dim, hidden_dim=intermediate_size)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[np.ndarray, np.ndarray]]]:
        # Pre-LN Residual stream: x + Attn(RMSNorm(x))
        norm_x = self.attn_norm(x)
        if kv_cache is not None:
            attn_out, new_cache = self.attn(norm_x, kv_cache=kv_cache)
        else:
            attn_out = self.attn(norm_x)
            new_cache = None

        x = x + attn_out

        # Second Pre-LN Residual stream: x + MLP(RMSNorm(x))
        norm_mlp_x = self.mlp_norm(x)
        mlp_out = self.mlp(norm_mlp_x)
        x = x + mlp_out

        if kv_cache is not None:
            return x, new_cache
        return x


@dataclass
class ParallaxLMConfig:
    vocab_size: int = 256
    embed_dim: int = 64
    num_heads: int = 4
    num_kv_heads: Optional[int] = None  # None = MHA, < num_heads = GQA
    num_layers: int = 3
    max_seq_len: int = 128
    use_rope: bool = True
    use_swiglu: bool = False
    tie_weights: bool = False
    num_experts: Optional[int] = None
    top_k_experts: int = 2
    intermediate_size: Optional[int] = None
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    rope_theta: float = 10000.0
    rope_interleaved: bool = True


class Embedding(Module):
    """Token Embedding lookup table."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Gaussian init with std=0.02
        embed_data = np.random.normal(0.0, 0.02, (vocab_size, embed_dim))
        self.weight = Parameter(embed_data, label="Embedding.weight")

    def forward(self, idx: Union[np.ndarray, Tensor]) -> Tensor:
        # idx: [batch_size, seq_len] array of integer token IDs
        idx_arr = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        B, T = idx_arr.shape
        out_data = self.weight.data[idx_arr]  # Shape: [B, T, embed_dim]

        out = Tensor(out_data, (self.weight,), "Embedding")

        def _backward() -> None:
            if self.weight.requires_grad and self.weight.grad is not None:
                # Accumulate gradients into indexed rows
                dout = out.grad
                np.add.at(self.weight.grad, idx_arr, dout)

        out._backward = _backward
        return out


class ParallaxLM(Module):
    """Decoder-only autoregressive language model."""

    def __init__(self, config: ParallaxLMConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.embed_dim)
        self.blocks = [
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                num_kv_heads=config.num_kv_heads,
                use_rope=config.use_rope,
                use_swiglu=config.use_swiglu,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=config.qkv_bias,
                rope_theta=config.rope_theta,
                rope_interleaved=config.rope_interleaved,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.lm_head = Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight Tying (Press & Wolf, 2016): share embedding and unembedding weights
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        idx: np.ndarray,
        kv_caches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        return_hidden: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[np.ndarray, np.ndarray]]], Tuple[Tensor, Tensor]]:
        # idx: [batch_size, seq_len]
        x = self.token_embeddings(idx)

        new_caches: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, block in enumerate(self.blocks):
            if kv_caches is not None:
                x, updated_cache = block(x, kv_cache=kv_caches[i])
                new_caches.append(updated_cache)
            else:
                x = block(x)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)  # [B, T, vocab_size]

        if return_hidden:
            return logits, hidden
        if kv_caches is not None:
            return logits, new_caches
        return logits
