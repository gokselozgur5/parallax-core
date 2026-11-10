"""
LLaMA model family implementation for ParallaxLM.

Provides LlamaConfig, LlamaForCausalLM, and HuggingFace Llama-3-8B compatibility.
Zero-dependency implementation using NumPy and Parallax primitives.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.tensor import Tensor


@dataclass
class LlamaConfig(ParallaxLMConfig):
    """Configuration class for LLaMA family models (LLaMA, LLaMA-2, LLaMA-3)."""

    vocab_size: int = 128256
    embed_dim: int = 4096
    num_heads: int = 32
    num_kv_heads: Optional[int] = 8
    num_layers: int = 32
    max_seq_len: int = 8192
    use_rope: bool = True
    use_swiglu: bool = True
    tie_weights: bool = False
    intermediate_size: Optional[int] = 14336
    rms_norm_eps: float = 1e-5
    qkv_bias: bool = False
    rope_theta: float = 500000.0
    rope_interleaved: bool = False  # HF standard rotary layout
    use_flash_attn: bool = False

    @classmethod
    def llama_3_8b(cls, **kwargs) -> "LlamaConfig":
        """Returns standard configuration for Meta-Llama-3-8B."""
        cfg_dict = {
            "vocab_size": 128256,
            "embed_dim": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "num_layers": 32,
            "max_seq_len": 8192,
            "use_rope": True,
            "use_swiglu": True,
            "tie_weights": False,
            "intermediate_size": 14336,
            "rms_norm_eps": 1e-5,
            "qkv_bias": False,
            "rope_theta": 500000.0,
            "rope_interleaved": False,
        }
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def llama_2_7b(cls, **kwargs) -> "LlamaConfig":
        """Returns standard configuration for Llama-2-7B."""
        cfg_dict = {
            "vocab_size": 32000,
            "embed_dim": 4096,
            "num_heads": 32,
            "num_kv_heads": 32,
            "num_layers": 32,
            "max_seq_len": 4096,
            "use_rope": True,
            "use_swiglu": True,
            "tie_weights": False,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "rope_interleaved": False,
        }
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)


class LlamaForCausalLM(ParallaxLM):
    """LlamaForCausalLM compatible with HuggingFace LLaMA-3 architecture.

    Zero-dependency autoregressive model with Grouped-Query Attention (GQA),
    SwiGLU feed-forward network, RoPE with configurable base theta (500k for LLaMA-3),
    and RMSNorm.
    """

    def __init__(self, config: Optional[LlamaConfig] = None):
        config = config or LlamaConfig.llama_3_8b()
        super().__init__(config)
