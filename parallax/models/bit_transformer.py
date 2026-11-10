"""
BitTransformer: Pure 1.58-bit Ternary Transformer Language Model Architecture.

Fuses BitNet b1.58 ternary linear projections {-1, 0, +1} (Ma et al., 2024)
across all attention projections (BitAttention), feed-forward SwiGLU MLP layers (BitSwiGLU),
and language model prediction heads (BitLinear LM Head). Replaces ~95%+ of floating-point
matrix multiplications with integer addition-dominated matrix products while supporting
exact straight-through estimator (STE) gradient propagation and KV-cache decoding.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union
import numpy as np

from parallax.bit_attention import BitAttention
from parallax.bitnet import BitLinear
from parallax.generator import sample_next_token
from parallax.models.gpt import Embedding, tensor_silu
from parallax.nn import Module, RMSNorm
from parallax.tensor import Tensor, no_grad


class BitSwiGLU(Module):
    """Feed-Forward Network using 1.58-bit BitLinear projections and SiLU activation.

    out = (SiLU(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T
    where W_gate, W_up, and W_down are BitLinear layers with ternary weights {-1, 0, 1}.
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * embed_dim) / 3)
            hidden_dim = 64 * ((hidden_dim + 64 - 1) // 64)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.w_gate = BitLinear(embed_dim, hidden_dim, bias=False)
        self.w_up = BitLinear(embed_dim, hidden_dim, bias=False)
        self.w_down = BitLinear(hidden_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = tensor_silu(self.w_gate(x))
        up = self.w_up(x)
        fused = gate * up
        out = self.w_down(fused)
        return out


class BitTransformerBlock(Module):
    """Pre-LN 1.58-bit Transformer Decoder Block.

    Architecture:
    x_norm1 = RMSNorm(x)
    attn_out = BitAttention(x_norm1)
    x = x + attn_out
    x_norm2 = RMSNorm(x)
    mlp_out = BitSwiGLU(x_norm2)
    x = x + mlp_out
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        use_rope: bool = True,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_interleaved: bool = True,
        use_flash_attn: bool = False,
        sliding_window: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.attn = BitAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            use_rope=use_rope,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
            rope_interleaved=rope_interleaved,
            use_flash_attn=use_flash_attn,
            sliding_window=sliding_window,
        )
        self.mlp_norm = RMSNorm(embed_dim, eps=rms_norm_eps)
        self.mlp = BitSwiGLU(embed_dim=embed_dim, hidden_dim=intermediate_size)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        rope_offset: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[np.ndarray, np.ndarray]]]:
        # Pre-LN BitAttention stream
        norm_x = self.attn_norm(x)
        if kv_cache is not None:
            attn_out, new_cache = self.attn(norm_x, kv_cache=kv_cache, rope_offset=rope_offset)
        else:
            attn_out = self.attn(norm_x, rope_offset=rope_offset)
            new_cache = None

        x = x + attn_out

        # Pre-LN BitSwiGLU stream
        norm_mlp_x = self.mlp_norm(x)
        mlp_out = self.mlp(norm_mlp_x)
        x = x + mlp_out

        if kv_cache is not None:
            return x, new_cache
        return x


@dataclass
class BitTransformerConfig:
    """Configuration for BitTransformerLM."""

    vocab_size: int = 256
    embed_dim: int = 64
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    num_layers: int = 3
    max_seq_len: int = 128
    intermediate_size: Optional[int] = None
    use_rope: bool = True
    qkv_bias: bool = False
    rope_theta: float = 10000.0
    rope_interleaved: bool = True
    use_flash_attn: bool = False
    sliding_window: Optional[int] = None
    rms_norm_eps: float = 1e-6
    tie_weights: bool = False


class BitTransformerLM(Module):
    """Pure 1.58-bit Causal Transformer Language Model.

    Combines:
    1. Token Embedding lookup
    2. N stacked BitTransformerBlock layers (BitAttention + BitSwiGLU)
    3. Final RMSNorm
    4. BitLinear LM Head (1.58-bit ternary projection to vocabulary)
    """

    def __init__(self, config: BitTransformerConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.embed_dim)
        self.blocks = [
            BitTransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_size=config.intermediate_size,
                use_rope=config.use_rope,
                qkv_bias=config.qkv_bias,
                rope_theta=config.rope_theta,
                rope_interleaved=config.rope_interleaved,
                use_flash_attn=config.use_flash_attn,
                sliding_window=config.sliding_window,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.lm_head = BitLinear(config.embed_dim, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        idx: Union[np.ndarray, Tensor],
        kv_caches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        return_hidden: bool = False,
        rope_offset: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[np.ndarray, np.ndarray]]], Tuple[Tensor, Tensor]]:
        """Forward pass over sequence or incremental token step.

        Parameters
        ----------
        idx : array_like or Tensor of token IDs with shape [B, T]
        kv_caches : Optional list of (past_k, past_v) tuples per layer for incremental decoding
        return_hidden : If True, return (logits, final_hidden_state)
        rope_offset : Optional starting sequence offset for RoPE embeddings
        """
        x = self.token_embeddings(idx)

        new_caches: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, block in enumerate(self.blocks):
            if kv_caches is not None:
                x, updated_cache = block(x, kv_cache=kv_caches[i], rope_offset=rope_offset)
                new_caches.append(updated_cache)
            else:
                x = block(x, rope_offset=rope_offset)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)  # [B, T, vocab_size]

        if return_hidden:
            return logits, hidden
        if kv_caches is not None:
            return logits, new_caches
        return logits

    def ternary_ratio(self) -> float:
        """Calculates the exact ratio of ternary parameters (BitLinear layers) across the model."""
        ternary_params = 0
        total_params = 0

        # Modules with BitLinear weights
        for block in self.blocks:
            # BitAttention projections: q_proj, k_proj, v_proj, out_proj
            for proj in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj, block.attn.out_proj]:
                ternary_params += proj.weight.data.size
                total_params += proj.weight.data.size
                if proj.bias is not None:
                    total_params += proj.bias.data.size
            # BitSwiGLU projections: w_gate, w_up, w_down
            for proj in [block.mlp.w_gate, block.mlp.w_up, block.mlp.w_down]:
                ternary_params += proj.weight.data.size
                total_params += proj.weight.data.size
                if proj.bias is not None:
                    total_params += proj.bias.data.size

            # Norm scales
            total_params += block.attn_norm.weight.data.size
            total_params += block.mlp_norm.weight.data.size

        # Embedding & final norm & lm_head
        total_params += self.token_embeddings.weight.data.size
        total_params += self.final_norm.weight.data.size
        ternary_params += self.lm_head.weight.data.size
        total_params += self.lm_head.weight.data.size

        return float(ternary_params / total_params) if total_params > 0 else 0.0

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Autoregressively generates new tokens conditioned on prompt tokens."""
        tokens = list(prompt_tokens)
        for token in self.generate_stream(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
            eos_token_id=eos_token_id,
        ):
            tokens.append(token)
        return tokens

    def generate_stream(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 10,
        top_p: Optional[float] = None,
        use_cache: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Iterator[int]:
        """Yields newly generated token IDs one by one using KV caching or sequence recompute."""
        from parallax.generator import sample_next_token

        tokens = list(prompt_tokens)
        max_seq_len = self.config.max_seq_len

        if not use_cache:
            for _ in range(max_new_tokens):
                context = tokens[-max_seq_len:]
                idx = np.array([context], dtype=np.int64)
                with no_grad():
                    logits = self.forward(idx)
                next_token_logits = logits.data[0, -1, :]
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                tokens.append(next_token)
                yield next_token
                if eos_token_id is not None and next_token == eos_token_id:
                    break
            return

        # Fast KV-cached autoregressive generation
        context = tokens[-max_seq_len:]
        idx = np.array([context], dtype=np.int64)

        B = 1
        num_heads = self.config.num_heads
        num_kv_heads = (
            self.config.num_kv_heads
            if self.config.num_kv_heads is not None
            else num_heads
        )
        head_dim = self.config.embed_dim // num_heads

        kv_caches: List[Tuple[np.ndarray, np.ndarray]] = [
            (
                np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
                np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
            )
            for _ in range(self.config.num_layers)
        ]

        with no_grad():
            logits, kv_caches = self.forward(idx, kv_caches=kv_caches)

        next_token_logits = logits.data[0, -1, :]
        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tokens.append(next_token)
        yield next_token

        if eos_token_id is not None and next_token == eos_token_id:
            return

        for _ in range(max_new_tokens - 1):
            if len(context) >= max_seq_len:
                context = tokens[-max_seq_len:]
                idx = np.array([context], dtype=np.int64)
                kv_caches = [
                    (
                        np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
                        np.empty((B, num_kv_heads, 0, head_dim), dtype=np.float64),
                    )
                    for _ in range(self.config.num_layers)
                ]
                with no_grad():
                    logits, kv_caches = self.forward(idx, kv_caches=kv_caches)
            else:
                context.append(next_token)
                idx_next = np.array([[next_token]], dtype=np.int64)
                with no_grad():
                    logits, kv_caches = self.forward(idx_next, kv_caches=kv_caches)

            next_token_logits = logits.data[0, -1, :]
            next_token = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens.append(next_token)
            yield next_token

            if eos_token_id is not None and next_token == eos_token_id:
                break
