"""
Zero-dependency GGUF v3 Model Exporter and Quantization Pipeline.

Implements end-to-end serialization of Parallax models (ParallaxLM / GPT architectures,
BitMamba SSM) into official GGUF v3 format compatible with llama.cpp and Ollama.
Supports uncompressed F32/F16 as well as Q8_0 and Q4_0 block quantization.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from parallax.bitmamba import BitMambaLM
from parallax.gguf import (
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q8_0,
    LLAMA_FTYPE_ALL_F32,
    LLAMA_FTYPE_MOSTLY_F16,
    LLAMA_FTYPE_MOSTLY_Q4_0,
    LLAMA_FTYPE_MOSTLY_Q8_0,
    save_gguf,
)
from parallax.gguf_quant import quantize_q4_0, quantize_q8_0
from parallax.models.gpt import ParallaxLM


def _collect_gpt_tensors(model: ParallaxLM) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Extracts metadata and tensor map from a ParallaxLM model."""
    cfg = model.config
    arch = "llama"

    if (
        hasattr(cfg, "num_experts")
        and cfg.num_experts is not None
        and cfg.num_experts > 1
    ) or any(
        hasattr(block, "moe")
        or (hasattr(block, "mlp") and (hasattr(block.mlp, "router") or hasattr(block.mlp, "experts")))
        for block in model.blocks
    ):
        raise NotImplementedError("MoE model export is not supported.")

    if cfg.intermediate_size is not None:
        intermediate_size = cfg.intermediate_size
    elif cfg.use_swiglu:
        intermediate_size = int(2 * (4 * cfg.embed_dim) / 3)
        intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
    else:
        intermediate_size = 4 * cfg.embed_dim

    num_kv_heads = cfg.num_kv_heads if cfg.num_kv_heads is not None else cfg.num_heads

    metadata: Dict[str, Any] = {
        "general.architecture": arch,
        "general.name": "ParallaxLM",
        f"{arch}.embedding_length": int(cfg.embed_dim),
        f"{arch}.block_count": int(cfg.num_layers),
        f"{arch}.feed_forward_length": int(intermediate_size),
        f"{arch}.attention.head_count": int(cfg.num_heads),
        f"{arch}.attention.head_count_kv": int(num_kv_heads),
        f"{arch}.context_length": int(cfg.max_seq_len),
        f"{arch}.attention.layer_norm_rms_epsilon": float(cfg.rms_norm_eps),
        f"{arch}.rope.freq_base": float(cfg.rope_theta),
        f"{arch}.rope.interleaved": bool(cfg.rope_interleaved),
        f"{arch}.use_swiglu": bool(cfg.use_swiglu),
    }

    tensors: Dict[str, np.ndarray] = {
        "token_embd.weight": model.token_embeddings.weight.data.astype(np.float32),
        "output_norm.weight": model.final_norm.weight.data.astype(np.float32),
    }

    if not cfg.tie_weights:
        tensors["output.weight"] = model.lm_head.weight.data.astype(np.float32)

    for i in range(cfg.num_layers):
        block = model.blocks[i]
        tensors[f"blk.{i}.attn_norm.weight"] = block.attn_norm.weight.data.astype(np.float32)
        tensors[f"blk.{i}.ffn_norm.weight"] = block.mlp_norm.weight.data.astype(np.float32)

        tensors[f"blk.{i}.attn_q.weight"] = block.attn.q_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_k.weight"] = block.attn.k_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_v.weight"] = block.attn.v_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.attn_output.weight"] = block.attn.out_proj.weight.data.astype(np.float32)

        if block.attn.q_proj.bias is not None:
            tensors[f"blk.{i}.attn_q.bias"] = block.attn.q_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_q.bias"] = np.zeros(block.attn.q_proj.out_features, dtype=np.float32)

        if block.attn.k_proj.bias is not None:
            tensors[f"blk.{i}.attn_k.bias"] = block.attn.k_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_k.bias"] = np.zeros(block.attn.k_proj.out_features, dtype=np.float32)

        if block.attn.v_proj.bias is not None:
            tensors[f"blk.{i}.attn_v.bias"] = block.attn.v_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"blk.{i}.attn_v.bias"] = np.zeros(block.attn.v_proj.out_features, dtype=np.float32)

        if block.attn.out_proj.bias is not None:
            tensors[f"blk.{i}.attn_output.bias"] = block.attn.out_proj.bias.data.astype(np.float32)

        if hasattr(block.mlp, "w_gate"):
            tensors[f"blk.{i}.ffn_gate.weight"] = block.mlp.w_gate.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_up.weight"] = block.mlp.w_up.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_down.weight"] = block.mlp.w_down.weight.data.astype(np.float32)
        elif hasattr(block.mlp, "fc1"):
            tensors[f"blk.{i}.ffn_up.weight"] = block.mlp.fc1.weight.data.astype(np.float32)
            tensors[f"blk.{i}.ffn_down.weight"] = block.mlp.fc2.weight.data.astype(np.float32)
            if block.mlp.fc1.bias is not None:
                tensors[f"blk.{i}.ffn_up.bias"] = block.mlp.fc1.bias.data.astype(np.float32)
            if block.mlp.fc2.bias is not None:
                tensors[f"blk.{i}.ffn_down.bias"] = block.mlp.fc2.bias.data.astype(np.float32)

    return metadata, tensors


def _collect_bitmamba_tensors(model: BitMambaLM) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Extracts metadata and tensor map from a BitMambaLM model."""
    cfg = model.config
    arch = "mamba"

    metadata: Dict[str, Any] = {
        "general.architecture": arch,
        "general.name": "BitMambaLM",
        "general.version": "2",
        f"{arch}.version": 2,
        f"{arch}.embedding_length": int(cfg.d_model),
        f"{arch}.block_count": int(cfg.num_layers),
        f"{arch}.ssm.d_state": int(cfg.d_state),
        f"{arch}.ssm.d_conv": int(cfg.d_conv),
        f"{arch}.ssm.expand": int(cfg.expand),
        f"{arch}.ssm.dt_rank": int(max(1, cfg.d_model // 16)),
        f"{arch}.attention.layer_norm_rms_epsilon": float(cfg.rms_norm_eps),
    }

    tensors: Dict[str, np.ndarray] = {
        "token_embd.weight": model.token_embeddings.weight.data.astype(np.float32),
        "output_norm.weight": model.final_norm.weight.data.astype(np.float32),
    }

    if not cfg.tie_weights:
        tensors["output.weight"] = model.lm_head.weight.data.astype(np.float32)

    for i in range(cfg.num_layers):
        block = model.blocks[i]
        tensors[f"blk.{i}.attn_norm.weight"] = model.norms[i].weight.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_in.weight"] = block.in_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_conv1d.weight"] = block.conv_weight.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_conv1d.bias"] = block.conv_bias.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_x.weight"] = block.x_proj.weight.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_dt.weight"] = block.dt_proj.weight.data.astype(np.float32)
        if block.dt_proj.bias is not None:
            tensors[f"blk.{i}.ssm_dt.bias"] = block.dt_proj.bias.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_a"] = block.A_log.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_d"] = block.D.data.astype(np.float32)
        tensors[f"blk.{i}.ssm_out.weight"] = block.out_proj.weight.data.astype(np.float32)

    return metadata, tensors


def export_to_gguf_v3(
    model: Union[ParallaxLM, BitMambaLM],
    file_path: str,
    quantization: str = "f32",
    tokenizer: Optional[Any] = None,
    architecture: Optional[str] = None,
) -> None:
    """Exports a ParallaxLM (GPT) or BitMambaLM model into an official GGUF v3 file.

    Parameters
    ----------
    model : ParallaxLM or BitMambaLM instance
    file_path : destination path for the .gguf binary
    quantization : quantization format, one of ("f32", "f16", "q8_0", "q4_0")
    tokenizer : optional tokenizer or GGUFTokenizer providing vocab and special token ids
    architecture : override architecture tag (defaults to "llama" for ParallaxLM, "mamba" for BitMambaLM)
    """
    quant = quantization.lower().strip()
    valid_quants = ("f32", "f16", "q8_0", "q4_0")
    if quant not in valid_quants:
        raise ValueError(f"Unsupported quantization: '{quantization}'. Expected one of {valid_quants}")

    # Extract metadata and tensors depending on model type
    if isinstance(model, BitMambaLM):
        metadata, tensors = _collect_bitmamba_tensors(model)
        vocab_size = model.config.vocab_size
    elif isinstance(model, ParallaxLM):
        metadata, tensors = _collect_gpt_tensors(model)
        vocab_size = model.config.vocab_size
    else:
        raise TypeError(f"Unsupported model type for GGUF export: {type(model).__name__}")

    if architecture is not None:
        metadata["general.architecture"] = architecture.lower()

    # Populate tokenizer metadata
    if tokenizer is not None:
        if hasattr(tokenizer, "raw_tokens"):
            metadata["tokenizer.ggml.tokens"] = list(tokenizer.raw_tokens)
        elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
            sorted_tokens = [tok for tok, _ in sorted(tokenizer.vocab.items(), key=lambda x: x[1])]
            metadata["tokenizer.ggml.tokens"] = sorted_tokens
        if getattr(tokenizer, "bos_token_id", None) is not None:
            metadata["tokenizer.ggml.bos_token_id"] = int(tokenizer.bos_token_id)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            metadata["tokenizer.ggml.eos_token_id"] = int(tokenizer.eos_token_id)
    else:
        default_tokens = []
        for i in range(vocab_size):
            if i < 256:
                default_tokens.append(f"<0x{i:02X}>")
            else:
                default_tokens.append(f"<tok_{i}>")
        metadata["tokenizer.ggml.tokens"] = default_tokens

    # Quantize tensors if requested
    tensor_payloads: Dict[str, Union[np.ndarray, bytes]] = {}
    tensor_types: Dict[str, int] = {}
    tensor_shapes: Dict[str, Tuple[int, ...]] = {}

    if quant == "f32":
        metadata["general.file_type"] = LLAMA_FTYPE_ALL_F32
        for name, arr in tensors.items():
            tensor_payloads[name] = arr.astype(np.float32)
            tensor_types[name] = GGML_TYPE_F32
            tensor_shapes[name] = arr.shape
    elif quant == "f16":
        metadata["general.file_type"] = LLAMA_FTYPE_MOSTLY_F16
        for name, arr in tensors.items():
            tensor_payloads[name] = arr.astype(np.float16)
            tensor_types[name] = GGML_TYPE_F16
            tensor_shapes[name] = arr.shape
    else:
        # Block quantization (Q8_0 or Q4_0)
        if quant == "q8_0":
            target_type = GGML_TYPE_Q8_0
            metadata["general.file_type"] = LLAMA_FTYPE_MOSTLY_Q8_0
            quant_fn = quantize_q8_0
        else:
            target_type = GGML_TYPE_Q4_0
            metadata["general.file_type"] = LLAMA_FTYPE_MOSTLY_Q4_0
            quant_fn = quantize_q4_0

        metadata["general.quantization_version"] = 2

        for name, arr in tensors.items():
            # Only quantize 2D weight matrices whose inner dimension is multiple of 32
            # 1D norms, biases and non-multiples remain F32
            if arr.ndim >= 2 and arr.shape[-1] % 32 == 0 and "norm" not in name:
                q_bytes = quant_fn(arr)
                tensor_payloads[name] = q_bytes
                tensor_types[name] = target_type
                tensor_shapes[name] = arr.shape
            else:
                tensor_payloads[name] = arr.astype(np.float32)
                tensor_types[name] = GGML_TYPE_F32
                tensor_shapes[name] = arr.shape

    save_gguf(
        file_path,
        metadata=metadata,
        tensors=tensor_payloads,
        version=3,
        tensor_types=tensor_types,
        tensor_shapes=tensor_shapes,
    )
