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
from parallax.models.llama import LlamaConfig, LlamaForCausalLM
from parallax.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    calculate_qwen2_intermediate_size,
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
    apply_pruning_masks,
    calculate_model_sparsity,
    create_pruning_step_hook,
    prune_2_to_4_structured,
    prune_model,
    prune_unstructured_magnitude,
)
from parallax.attention_sink import RollingKVCache
from parallax.kv_cache_pruner import KVCachePruner, PruningMetrics
from parallax.linear_attention import LinearAttention, elu_feature_map
from parallax.bitmamba import BitMambaBlock, BitMambaConfig, BitMambaLM
from parallax.bit_attention import BitAttention

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
    "LlamaConfig",
    "LlamaForCausalLM",
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "calculate_qwen2_intermediate_size",
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
    "apply_pruning_masks",
    "create_pruning_step_hook",
    "RollingKVCache",
    "LinearAttention",
    "elu_feature_map",
    "BitMambaBlock",
    "BitMambaConfig",
    "BitMambaLM",
    "BitAttention",
    "compute_cross_entropy_loss",
    "compute_perplexity",
    "ContinuousBatchEngine",
    "RequestStatus",
    "SequenceRequest",
]
