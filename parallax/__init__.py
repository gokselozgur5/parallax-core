"""Parallax Deep Learning Framework."""

from parallax.batching import ContinuousBatchEngine, RequestStatus, SequenceRequest
from parallax.tensor import Tensor, is_grad_enabled, no_grad
from parallax.eval import compute_cross_entropy_loss, compute_perplexity
from parallax.generator import generate, generate_stream
from parallax.safetensors import load_safetensors, save_safetensors
from parallax.hf_loader import (
    hf_config_to_parallax,
    load_hf_pretrained,
    load_hf_weights_into_model,
    save_hf_pretrained,
)
from parallax.hf_tokenizer import HFTokenizer, load_hf_tokenizer
from parallax.gguf import GGUFFile, GGUFTensorInfo, load_gguf, save_gguf
from parallax.gguf_loader import (
    gguf_metadata_to_parallax,
    load_gguf_pretrained,
    load_gguf_weights_into_model,
    save_parallax_to_gguf,
)
from parallax.gguf_tokenizer import GGUFTokenizer, load_gguf_tokenizer
from parallax.gguf_quant import (
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from parallax.prune import (
    calculate_model_sparsity,
    prune_2_to_4_structured,
    prune_model,
    prune_unstructured_magnitude,
)
from parallax.attention_sink import RollingKVCache
from parallax.linear_attention import LinearAttention, elu_feature_map

__all__ = [
    "Tensor",
    "is_grad_enabled",
    "no_grad",
    "generate",
    "generate_stream",
    "load_safetensors",
    "save_safetensors",
    "hf_config_to_parallax",
    "load_hf_pretrained",
    "load_hf_weights_into_model",
    "save_hf_pretrained",
    "HFTokenizer",
    "load_hf_tokenizer",
    "GGUFFile",
    "GGUFTensorInfo",
    "load_gguf",
    "save_gguf",
    "gguf_metadata_to_parallax",
    "load_gguf_pretrained",
    "load_gguf_weights_into_model",
    "save_parallax_to_gguf",
    "GGUFTokenizer",
    "load_gguf_tokenizer",
    "dequantize_q4_0",
    "dequantize_q8_0",
    "quantize_q4_0",
    "quantize_q8_0",
    "prune_unstructured_magnitude",
    "prune_2_to_4_structured",
    "calculate_model_sparsity",
    "prune_model",
    "RollingKVCache",
    "LinearAttention",
    "elu_feature_map",
    "compute_cross_entropy_loss",
    "compute_perplexity",
    "ContinuousBatchEngine",
    "RequestStatus",
    "SequenceRequest",
]
