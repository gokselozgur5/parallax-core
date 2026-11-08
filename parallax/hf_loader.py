"""
Zero-dependency HuggingFace Model Loader for ParallaxLM.

Loads pretrained HuggingFace models (LLaMA, SmolLM, Qwen2, Mistral)
from pure SafeTensors binary checkpoints without requiring `torch`
or `transformers`.
"""

import json
import os
from typing import Any, Dict, Optional
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Parameter
from parallax.safetensors import load_safetensors


def _assign_param(param: Parameter, hf_tensors: Dict[str, np.ndarray], key: str) -> None:
    if key not in hf_tensors:
        return
    src = hf_tensors[key]
    if param.data.shape != src.shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {param.data.shape}, got {src.shape}"
        )
    param.data = src.copy()
    if param.grad is not None and param.grad.shape != param.data.shape:
        param.grad = np.zeros_like(param.data)


def _assign_bias(proj: Linear, hf_tensors: Dict[str, np.ndarray], key: str) -> None:
    if key not in hf_tensors:
        return
    src = hf_tensors[key]
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


def hf_config_to_parallax(hf_config: Dict[str, Any]) -> ParallaxLMConfig:
    """Translates a HuggingFace model config.json into a ParallaxLMConfig."""
    embed_dim = int(hf_config.get("hidden_size", 576))
    num_heads = int(hf_config.get("num_attention_heads", 9))
    num_kv_heads = hf_config.get("num_key_value_heads")
    if num_kv_heads is not None:
        num_kv_heads = int(num_kv_heads)
    else:
        num_kv_heads = num_heads

    num_layers = int(hf_config.get("num_hidden_layers", 30))
    vocab_size = int(hf_config.get("vocab_size", 49152))
    max_seq_len = int(hf_config.get("max_position_embeddings", 2048))

    intermediate_size = hf_config.get("intermediate_size")
    if intermediate_size is not None:
        intermediate_size = int(intermediate_size)

    rms_norm_eps = float(hf_config.get("rms_norm_eps", 1e-5))
    rope_theta = float(hf_config.get("rope_theta", 10000.0))

    # Qwen2 and some architectures use attention_bias / qkv_bias
    model_type = str(hf_config.get("model_type", "")).lower()
    qkv_bias = bool(
        hf_config.get("attention_bias", False)
        or hf_config.get("qkv_bias", False)
        or model_type.startswith("qwen2")
    )

    tie_weights = bool(hf_config.get("tie_word_embeddings", True))

    return ParallaxLMConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=rope_theta,
        rope_interleaved=False,  # HF standard: [:d/2], [d/2:] split
        use_swiglu=True,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        tie_weights=tie_weights,
        qkv_bias=qkv_bias,
    )


def load_hf_weights_into_model(
    model: ParallaxLM,
    hf_tensors: Dict[str, np.ndarray],
) -> ParallaxLM:
    """Maps HuggingFace tensor state dictionary directly into a ParallaxLM model."""
    # 1. Token Embeddings
    _assign_param(model.token_embeddings.weight, hf_tensors, "model.embed_tokens.weight")

    # 2. Transformer Blocks
    for i in range(model.config.num_layers):
        prefix = f"model.layers.{i}"
        block = model.blocks[i]

        # LayerNorms
        _assign_param(block.attn_norm.weight, hf_tensors, f"{prefix}.input_layernorm.weight")
        _assign_param(block.mlp_norm.weight, hf_tensors, f"{prefix}.post_attention_layernorm.weight")

        # Attention weights
        _assign_param(block.attn.q_proj.weight, hf_tensors, f"{prefix}.self_attn.q_proj.weight")
        _assign_param(block.attn.k_proj.weight, hf_tensors, f"{prefix}.self_attn.k_proj.weight")
        _assign_param(block.attn.v_proj.weight, hf_tensors, f"{prefix}.self_attn.v_proj.weight")
        _assign_param(block.attn.out_proj.weight, hf_tensors, f"{prefix}.self_attn.o_proj.weight")

        # Attention biases (if present, e.g. Qwen2)
        _assign_bias(block.attn.q_proj, hf_tensors, f"{prefix}.self_attn.q_proj.bias")
        _assign_bias(block.attn.k_proj, hf_tensors, f"{prefix}.self_attn.k_proj.bias")
        _assign_bias(block.attn.v_proj, hf_tensors, f"{prefix}.self_attn.v_proj.bias")
        _assign_bias(block.attn.out_proj, hf_tensors, f"{prefix}.self_attn.o_proj.bias")

        # SwiGLU MLP weights
        if hasattr(block.mlp, "w_gate"):
            _assign_param(block.mlp.w_gate.weight, hf_tensors, f"{prefix}.mlp.gate_proj.weight")
        if hasattr(block.mlp, "w_up"):
            _assign_param(block.mlp.w_up.weight, hf_tensors, f"{prefix}.mlp.up_proj.weight")
        if hasattr(block.mlp, "w_down"):
            _assign_param(block.mlp.w_down.weight, hf_tensors, f"{prefix}.mlp.down_proj.weight")

    # 3. Final Norm
    _assign_param(model.final_norm.weight, hf_tensors, "model.norm.weight")

    # 4. LM Head
    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    else:
        _assign_param(model.lm_head.weight, hf_tensors, "lm_head.weight")

    return model


def load_hf_pretrained(model_dir: str) -> ParallaxLM:
    """Loads a pretrained HuggingFace model directory into a ready ParallaxLM instance."""
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    # Check for single-file safetensors
    weights_file = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights_file):
        # Look for any .safetensors file in directory
        sf_candidates = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        if not sf_candidates:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
        weights_file = os.path.join(model_dir, sf_candidates[0])

    p_config = hf_config_to_parallax(hf_config)
    model = ParallaxLM(p_config)

    tensors = load_safetensors(weights_file, cast_float64=True)
    load_hf_weights_into_model(model, tensors)

    return model
