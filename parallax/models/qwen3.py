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

    @classmethod
    def qwen3_0_6b(cls, **kwargs: Any) -> "Qwen3Config":
        """Returns standard configuration for Qwen/Qwen3-0.6B."""
        cfg: Dict[str, Any] = {
            "vocab_size": 151936,
            "embed_dim": 1024,
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "num_layers": 28,
            "max_seq_len": 40960,
            "tie_weights": True,
            "intermediate_size": 3072,
        }
        cfg.update(kwargs)
        return cls(**cfg)

    @classmethod
    def qwen3_32b(cls, **kwargs: Any) -> "Qwen3Config":
        """Returns standard configuration for Qwen/Qwen3-32B."""
        cfg: Dict[str, Any] = {
            "vocab_size": 151936,
            "embed_dim": 5120,
            "num_heads": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
            "num_layers": 64,
            "max_seq_len": 40960,
            "tie_weights": False,
            "intermediate_size": 25600,
        }
        cfg.update(kwargs)
        return cls(**cfg)


class Qwen3ForCausalLM(ParallaxLM):
    """Qwen3ForCausalLM with GQA, per-head QK RMSNorm, and SwiGLU MLP."""

    def __init__(self, config: Optional[Qwen3Config] = None):
        super().__init__(config or Qwen3Config.qwen3_0_6b())
