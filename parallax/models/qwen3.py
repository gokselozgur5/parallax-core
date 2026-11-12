"""Qwen3 model family configuration and architecture for ParallaxLM."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


@dataclass
class Qwen3Config(ParallaxLMConfig):
    """Configuration for HuggingFace Qwen3 dense causal language models."""

    vocab_size: int = 151936
    embed_dim: int = 5120
    num_heads: int = 64
    num_kv_heads: Optional[int] = 8
    head_dim: Optional[int] = 128
    num_layers: int = 64
    max_seq_len: int = 40960
    use_rope: bool = True
    use_swiglu: bool = True
    tie_weights: bool = False
    intermediate_size: Optional[int] = 25600
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = False
    qk_norm: bool = True
    rope_theta: float = 1000000.0
    rope_interleaved: bool = False
