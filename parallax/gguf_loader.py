"""
Zero-dependency GGUF Model and Weights Loader for ParallaxLM.

Extracts architecture hyper-parameters and maps standard GGML/GGUF tensors
directly into a runnable ParallaxLM model without requiring `llama.cpp` or PyTorch.
"""

from typing import Any, Dict, Optional
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Parameter
from parallax.gguf import GGUFFile, load_gguf


def _assign_param(param: Parameter, tensors: Dict[str, np.ndarray], key: str) -> None:
    if key not in tensors:
        return
    src = tensors[key]
    if param.data.shape != src.shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {param.data.shape}, got {src.shape}"
        )
    param.data = src.copy()
    if param.grad is not None and param.grad.shape != param.data.shape:
        param.grad = np.zeros_like(param.data)


def _assign_bias(proj: Linear, tensors: Dict[str, np.ndarray], key: str) -> None:
    if key not in tensors:
        return
    src = tensors[key]
    expected_shape = (proj.out_features,)
    if src.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {expected_shape}, got {src.shape}"
        )
    if proj.bias is None:
        proj.bias = Parameter(src.copy(), label="Linear.bias")
    else:
        proj.bias.data = src.copy()
        if proj.bias.grad is not None and proj.bias.grad.shape != expected_shape:
            proj.bias.grad = np.zeros_like(proj.bias.data)


def gguf_metadata_to_parallax(
    metadata: Dict[str, Any],
    tensors: Optional[Dict[str, np.ndarray]] = None,
) -> ParallaxLMConfig:
    """Extracts ParallaxLMConfig from GGUF metadata dictionary."""
    arch = str(metadata.get("general.architecture", "llama")).lower()

    # Embedding and sequence dimensions
    embed_dim = int(metadata.get(f"{arch}.embedding_length", 576))
    num_layers = int(metadata.get(f"{arch}.block_count", 30))
    num_heads = int(metadata.get(f"{arch}.attention.head_count", 9))

    num_kv_heads = metadata.get(f"{arch}.attention.head_count_kv")
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)
    else:
        num_kv_heads = num_heads

    intermediate_size = metadata.get(f"{arch}.feed_forward_length")
    if intermediate_size is not None:
        intermediate_size = int(intermediate_size)

    max_seq_len = int(metadata.get(f"{arch}.context_length", 2048))
    rms_norm_eps = float(metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
    rope_theta = float(metadata.get(f"{arch}.rope.freq_base", 10000.0))

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

    return ParallaxLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=rope_theta,
        rope_interleaved=False,
        use_swiglu=True,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_weights=tie_weights,
        qkv_bias=qkv_bias,
    )


def load_gguf_weights_into_model(
    model: ParallaxLM,
    gguf_tensors: Dict[str, np.ndarray],
) -> ParallaxLM:
    """Maps standard GGUF tensor dictionary into ParallaxLM parameters."""
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

        # Attention projection biases (e.g. Qwen2)
        _assign_bias(block.attn.q_proj, gguf_tensors, f"blk.{i}.attn_q.bias")
        _assign_bias(block.attn.k_proj, gguf_tensors, f"blk.{i}.attn_k.bias")
        _assign_bias(block.attn.v_proj, gguf_tensors, f"blk.{i}.attn_v.bias")
        _assign_bias(block.attn.out_proj, gguf_tensors, f"blk.{i}.attn_output.bias")

        # SwiGLU MLP weights
        if hasattr(block.mlp, "w_gate"):
            _assign_param(block.mlp.w_gate.weight, gguf_tensors, f"blk.{i}.ffn_gate.weight")
        if hasattr(block.mlp, "w_up"):
            _assign_param(block.mlp.w_up.weight, gguf_tensors, f"blk.{i}.ffn_up.weight")
        if hasattr(block.mlp, "w_down"):
            _assign_param(block.mlp.w_down.weight, gguf_tensors, f"blk.{i}.ffn_down.weight")

    # 3. Final Norm
    _assign_param(model.final_norm.weight, gguf_tensors, "output_norm.weight")

    # 4. LM Head
    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    else:
        _assign_param(model.lm_head.weight, gguf_tensors, "output.weight")

    return model


def load_gguf_pretrained(file_path: str) -> ParallaxLM:
    """Parses a GGUF file and instantiates a ready-to-run ParallaxLM model."""
    gguf_file = load_gguf(file_path, cast_float64=True)
    p_config = gguf_metadata_to_parallax(gguf_file.metadata, gguf_file.tensors)
    model = ParallaxLM(p_config)
    load_gguf_weights_into_model(model, gguf_file.tensors)
    return model
