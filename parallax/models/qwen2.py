"""
Qwen2 and Qwen2.5 model family implementation for ParallaxLM.

Provides Qwen2Config, Qwen2ForCausalLM, and HuggingFace Qwen2.5 / Qwen2.5-Coder compatibility.
Zero-dependency implementation using NumPy and Parallax primitives.
"""

from dataclasses import dataclass
from typing import Optional

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def calculate_qwen2_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Calculates SwiGLU intermediate_size for Qwen2 architectures.

    Follows standard SwiGLU dimension scaling (~8/3 * hidden_size) rounded to
    the nearest hardware-friendly multiple.
    """
    raw_size = int(2 * (4 * hidden_size) / 3)
    return multiple_of * ((raw_size + multiple_of - 1) // multiple_of)


@dataclass
class Qwen2Config(ParallaxLMConfig):
    """Configuration class for Qwen2 and Qwen2.5 model families.

    Key architectural specifications:
    - Q/K/V bias is True, while output projection (o_proj) bias is False.
    - RMSNorm epsilon defaults to 1e-6.
    - RoPE base frequency (rope_theta) defaults to 1,000,000.0 (1M).
    - HF rotary layout: non-interleaved [:half, half:].
    - SwiGLU activation with SwiGLU feed-forward projections.
    - Sliding window attention (sliding_window) supported per layer.
    """

    vocab_size: int = 152064
    embed_dim: int = 3584
    num_heads: int = 28
    num_kv_heads: Optional[int] = 4
    num_layers: int = 28
    max_seq_len: int = 32768
    use_rope: bool = True
    use_swiglu: bool = True
    tie_weights: bool = False
    intermediate_size: Optional[int] = 18944
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = True
    rope_theta: float = 1000000.0
    rope_interleaved: bool = False
    use_flash_attn: bool = False
    sliding_window: Optional[int] = 32768

    @classmethod
    def qwen2_5_0_5b(cls, **kwargs) -> "Qwen2Config":
        """Returns standard configuration for Qwen/Qwen2.5-0.5B."""
        cfg_dict = {
            "vocab_size": 151936,
            "embed_dim": 896,
            "num_heads": 14,
            "num_kv_heads": 2,
            "num_layers": 24,
            "max_seq_len": 32768,
            "use_rope": True,
            "use_swiglu": True,
            "tie_weights": True,
            "intermediate_size": 4864,
            "rms_norm_eps": 1e-6,
            "qkv_bias": True,
            "rope_theta": 1000000.0,
            "rope_interleaved": False,
            "sliding_window": 32768,
        }
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def qwen2_5_coder_1_5b(cls, **kwargs) -> "Qwen2Config":
        """Returns standard configuration for Qwen/Qwen2.5-Coder-1.5B."""
        cfg_dict = {
            "vocab_size": 151936,
            "embed_dim": 1536,
            "num_heads": 12,
            "num_kv_heads": 2,
            "num_layers": 28,
            "max_seq_len": 32768,
            "use_rope": True,
            "use_swiglu": True,
            "tie_weights": True,
            "intermediate_size": 8960,
            "rms_norm_eps": 1e-6,
            "qkv_bias": True,
            "rope_theta": 1000000.0,
            "rope_interleaved": False,
            "sliding_window": 32768,
        }
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)

    @classmethod
    def qwen2_5_coder_7b(cls, **kwargs) -> "Qwen2Config":
        """Returns standard configuration for Qwen/Qwen2.5-Coder-7B."""
        cfg_dict = {
            "vocab_size": 152064,
            "embed_dim": 3584,
            "num_heads": 28,
            "num_kv_heads": 4,
            "num_layers": 28,
            "max_seq_len": 32768,
            "use_rope": True,
            "use_swiglu": True,
            "tie_weights": False,
            "intermediate_size": 18944,
            "rms_norm_eps": 1e-6,
            "qkv_bias": True,
            "rope_theta": 1000000.0,
            "rope_interleaved": False,
            "sliding_window": 131072,
        }
        cfg_dict.update(kwargs)
        return cls(**cfg_dict)


class Qwen2ForCausalLM(ParallaxLM):
    """Qwen2ForCausalLM compatible with HuggingFace Qwen2 / Qwen2.5 architectures.

    Zero-dependency autoregressive model with Grouped-Query Attention (GQA),
    Q/K/V bias support, SwiGLU feed-forward network, RoPE theta=1,000,000,
    RMSNorm epsilon=1e-6, and optional sliding window attention.
    """

    def __init__(self, config: Optional[Qwen2Config] = None):
        config = config or Qwen2Config.qwen2_5_coder_7b()
        super().__init__(config)
