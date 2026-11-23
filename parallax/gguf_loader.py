"""
Zero-dependency GGUF Model and Weights Loader for ParallaxLM.

Extracts architecture hyper-parameters and maps standard GGML/GGUF tensors
directly into a runnable ParallaxLM model without requiring `llama.cpp` or PyTorch.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.gguf import load_gguf
from parallax.gguf_tokenizer import GGUFTokenizer, load_gguf_tokenizer
from parallax.loader_utils import _assign_bias, _assign_param


def gguf_metadata_to_parallax(
    metadata: Dict[str, Any],
    tensors: Optional[Dict[str, np.ndarray]] = None,
) -> ParallaxLMConfig:
    """Extracts ParallaxLMConfig from GGUF metadata dictionary."""
    arch = str(metadata.get("general.architecture", "llama")).lower()
    if arch not in ("llama", "mistral", "qwen2", "qwen3", "smollm"):
        raise ValueError(
            f"Unsupported GGUF architecture: '{arch}'. Supported: llama, mistral, qwen2, qwen3, smollm."
        )

    # Architecture hyper-parameters
    required_keys = (
        f"{arch}.embedding_length",
        f"{arch}.block_count",
        f"{arch}.attention.head_count",
    )
    for req_key in required_keys:
        if req_key not in metadata:
            raise ValueError(f"Missing required GGUF metadata key: '{req_key}'")

    # Embedding and sequence dimensions
    embed_dim = int(metadata[f"{arch}.embedding_length"])
    num_layers = int(metadata[f"{arch}.block_count"])
    num_heads = int(metadata[f"{arch}.attention.head_count"])

    num_kv_heads = metadata.get(f"{arch}.attention.head_count_kv")
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)
    else:
        num_kv_heads = num_heads

    intermediate_size = metadata.get(f"{arch}.feed_forward_length")
    if intermediate_size is not None:
        intermediate_size = int(intermediate_size)

    max_seq_len = int(metadata.get(f"{arch}.context_length", 2048))
    default_eps = 1e-6 if arch.startswith(("qwen2", "qwen3")) else 1e-5
    default_theta = 1000000.0 if arch.startswith(("qwen2", "qwen3")) else 10000.0
    rms_norm_eps = float(metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", default_eps))
    rope_theta = float(metadata.get(f"{arch}.rope.freq_base", default_theta))

    # Vocab size calculation
    tokens = metadata.get("tokenizer.ggml.tokens")
    if tokens is not None:
        vocab_size = len(tokens)
    elif tensors and "token_embd.weight" in tensors:
        vocab_size = int(tensors["token_embd.weight"].shape[0])
    else:
        vocab_size = int(metadata.get("general.vocab_size", 32000))

    # Determine whether weights are tied (output.weight present vs omitted)
    if tensors is not None and "output.weight" in tensors:
        tie_weights = False
    else:
        tie_weights = True

    # Detect QKV bias presence
    qkv_bias = False
    if tensors is not None:
        qkv_bias = any(
            k.startswith("blk.0.attn_q.bias") or k.startswith("blk.0.attn_k.bias")
            for k in tensors
        )
    elif arch.startswith("qwen2"):
        qkv_bias = True

    # Detect SwiGLU vs standard MLP
    if f"{arch}.use_swiglu" in metadata:
        use_swiglu = bool(metadata[f"{arch}.use_swiglu"])
    elif tensors is not None:
        use_swiglu = "blk.0.ffn_gate.weight" in tensors
    else:
        use_swiglu = True

    # RoPE layout convention
    # When omitted in metadata:
    # - llama, mistral: llama.cpp permutes weights to interleaved layout (default: True)
    # - qwen2, phi, gemma, etc.: standard contiguous layout (default: False)
    if f"{arch}.rope.interleaved" in metadata:
        rope_interleaved = bool(metadata[f"{arch}.rope.interleaved"])
    else:
        rope_interleaved = arch in ("llama", "mistral")

    sliding_window = None
    if f"{arch}.attention.sliding_window" in metadata:
        sliding_window = int(metadata[f"{arch}.attention.sliding_window"])
    elif f"{arch}.sliding_window" in metadata:
        sliding_window = int(metadata[f"{arch}.sliding_window"])

    return ParallaxLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=rope_theta,
        rope_interleaved=rope_interleaved,
        use_swiglu=use_swiglu,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_weights=tie_weights,
        qkv_bias=qkv_bias,
        head_dim=int(metadata[f"{arch}.attention.key_length"]) if f"{arch}.attention.key_length" in metadata else None,
        qk_norm=bool(arch == "qwen3" or (tensors is not None and "blk.0.attn_q_norm.weight" in tensors)),
        sliding_window=sliding_window,
    )



def load_gguf_weights_into_model(
    model: ParallaxLM,
    gguf_tensors: Dict[str, np.ndarray],
    strict: bool = True,
) -> ParallaxLM:
    """Maps standard GGUF tensor dictionary into ParallaxLM parameters."""
    expected_keys = [
        "token_embd.weight",
        "output_norm.weight",
    ]
    if not model.config.tie_weights:
        expected_keys.append("output.weight")

    for i in range(model.config.num_layers):
        block = model.blocks[i]
        expected_keys.extend([
            f"blk.{i}.attn_norm.weight",
            f"blk.{i}.ffn_norm.weight",
            f"blk.{i}.attn_q.weight",
            f"blk.{i}.attn_k.weight",
            f"blk.{i}.attn_v.weight",
            f"blk.{i}.attn_output.weight",
        ])
        if model.config.qk_norm:
            expected_keys.extend([
                f"blk.{i}.attn_q_norm.weight",
                f"blk.{i}.attn_k_norm.weight",
            ])
        if model.config.qkv_bias:
            expected_keys.extend([
                f"blk.{i}.attn_q.bias",
                f"blk.{i}.attn_k.bias",
                f"blk.{i}.attn_v.bias",
            ])
            if getattr(block.attn, "out_proj", None) is not None and block.attn.out_proj.bias is not None:
                expected_keys.append(f"blk.{i}.attn_output.bias")
        if model.config.use_swiglu:
            expected_keys.extend([
                f"blk.{i}.ffn_gate.weight",
                f"blk.{i}.ffn_up.weight",
                f"blk.{i}.ffn_down.weight",
            ])
        else:
            expected_keys.extend([
                f"blk.{i}.ffn_up.weight",
                f"blk.{i}.ffn_down.weight",
            ])
            block = model.blocks[i]
            if hasattr(block.mlp, "fc1") and block.mlp.fc1.bias is not None:
                expected_keys.append(f"blk.{i}.ffn_up.bias")
            if hasattr(block.mlp, "fc2") and block.mlp.fc2.bias is not None:
                expected_keys.append(f"blk.{i}.ffn_down.bias")

    missing_keys = [k for k in expected_keys if k not in gguf_tensors]
    unexpected_keys = [k for k in gguf_tensors if k not in expected_keys]

    object.__setattr__(model, "_missing_keys", missing_keys)
    object.__setattr__(model, "_unexpected_keys", unexpected_keys)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Strict GGUF weight loading failed. Missing: {missing_keys}, Unexpected: {unexpected_keys}"
        )

    # 1. Token Embeddings
    _assign_param(model.token_embeddings.weight, gguf_tensors, "token_embd.weight")

    # 2. Transformer Blocks
    for i in range(model.config.num_layers):
        block = model.blocks[i]

        # Normalization layers
        _assign_param(block.attn_norm.weight, gguf_tensors, f"blk.{i}.attn_norm.weight")
        _assign_param(block.mlp_norm.weight, gguf_tensors, f"blk.{i}.ffn_norm.weight")

        # Attention projection weights
        _assign_param(block.attn.q_proj.weight, gguf_tensors, f"blk.{i}.attn_q.weight")
        _assign_param(block.attn.k_proj.weight, gguf_tensors, f"blk.{i}.attn_k.weight")
        _assign_param(block.attn.v_proj.weight, gguf_tensors, f"blk.{i}.attn_v.weight")
        _assign_param(block.attn.out_proj.weight, gguf_tensors, f"blk.{i}.attn_output.weight")
        if block.attn.q_norm is not None and block.attn.k_norm is not None:
            _assign_param(block.attn.q_norm.weight, gguf_tensors, f"blk.{i}.attn_q_norm.weight")
            _assign_param(block.attn.k_norm.weight, gguf_tensors, f"blk.{i}.attn_k_norm.weight")

        # Attention projection biases (e.g. Qwen2)
        _assign_bias(block.attn.q_proj, gguf_tensors, f"blk.{i}.attn_q.bias")
        _assign_bias(block.attn.k_proj, gguf_tensors, f"blk.{i}.attn_k.bias")
        _assign_bias(block.attn.v_proj, gguf_tensors, f"blk.{i}.attn_v.bias")
        _assign_bias(block.attn.out_proj, gguf_tensors, f"blk.{i}.attn_output.bias")

        # SwiGLU or standard MLP weights
        mlp: Any = block.mlp
        if hasattr(mlp, "w_gate") and f"blk.{i}.ffn_gate.weight" in gguf_tensors:
            _assign_param(mlp.w_gate.weight, gguf_tensors, f"blk.{i}.ffn_gate.weight")
            _assign_param(mlp.w_up.weight, gguf_tensors, f"blk.{i}.ffn_up.weight")
            _assign_param(mlp.w_down.weight, gguf_tensors, f"blk.{i}.ffn_down.weight")
        elif hasattr(mlp, "fc1"):
            _assign_param(mlp.fc1.weight, gguf_tensors, f"blk.{i}.ffn_up.weight")
            _assign_param(mlp.fc2.weight, gguf_tensors, f"blk.{i}.ffn_down.weight")
            _assign_bias(mlp.fc1, gguf_tensors, f"blk.{i}.ffn_up.bias")
            _assign_bias(mlp.fc2, gguf_tensors, f"blk.{i}.ffn_down.bias")

    # 3. Final Norm
    _assign_param(model.final_norm.weight, gguf_tensors, "output_norm.weight")

    # 4. LM Head
    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    else:
        _assign_param(model.lm_head.weight, gguf_tensors, "output.weight")

    return model


def load_gguf_pretrained(
    file_path: str,
    return_tokenizer: bool = False,
    strict: bool = True,
) -> Union[ParallaxLM, Tuple[ParallaxLM, GGUFTokenizer]]:
    """Parses a GGUF file and instantiates a ready-to-run ParallaxLM model, optionally returning its tokenizer."""
    gguf_file = load_gguf(file_path, cast_float64=True)
    p_config = gguf_metadata_to_parallax(gguf_file.metadata, gguf_file.tensors)
    if "token_embd.weight" in gguf_file.tensor_infos:
        embd_shape = gguf_file.tensor_infos["token_embd.weight"].shape
        if len(embd_shape) == 2 and (embd_shape[1] != p_config.embed_dim or embd_shape[0] != p_config.vocab_size):
            raise ValueError(f"Mismatch between metadata dims and token_embd.weight shape {embd_shape}")
    max_blocks = max(
        (int(k.split(".")[1]) + 1 for k in gguf_file.tensor_infos if k.startswith("blk.") and k.split(".")[1].isdigit()),
        default=0,
    )
    if max_blocks > 0 and p_config.num_layers != max_blocks:
        raise ValueError(f"Metadata block_count {p_config.num_layers} does not match tensor layers {max_blocks}")
    if p_config.num_heads <= 0 or p_config.num_layers <= 0 or p_config.embed_dim <= 0 or p_config.max_seq_len <= 0:
        raise ValueError("Invalid non-positive architecture dimensions in GGUF metadata")
    model = ParallaxLM(p_config)
    load_gguf_weights_into_model(model, gguf_file.tensors, strict=strict)

    if return_tokenizer:
        tokenizer = load_gguf_tokenizer(gguf_file)
        return model, tokenizer

    return model


def save_parallax_to_gguf(
    model: ParallaxLM,
    file_path: str,
    tokenizer: Optional[Any] = None,
    architecture: str = "llama",
) -> None:
    """Serializes a ParallaxLM model and optional tokenizer into a standard GGUF v3 file."""
    from parallax.gguf_exporter import export_to_gguf_v3

    export_to_gguf_v3(
        model,
        file_path,
        quantization="f32",
        tokenizer=tokenizer,
        architecture=architecture,
    )

