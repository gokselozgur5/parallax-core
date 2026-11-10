"""Models package for ParallaxLM."""

from parallax.models.bit_transformer import (
    BitSwiGLU,
    BitTransformerBlock,
    BitTransformerConfig,
    BitTransformerLM,
)
from parallax.models.gpt import MLP, ParallaxLM, ParallaxLMConfig, SwiGLU, TransformerBlock
from parallax.models.llama import LlamaConfig, LlamaForCausalLM
from parallax.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    calculate_qwen2_intermediate_size,
)

__all__ = [
    "BitSwiGLU",
    "BitTransformerBlock",
    "BitTransformerConfig",
    "BitTransformerLM",
    "ParallaxLM",
    "ParallaxLMConfig",
    "TransformerBlock",
    "MLP",
    "SwiGLU",
    "LlamaConfig",
    "LlamaForCausalLM",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "calculate_qwen2_intermediate_size",
]
