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
from parallax.safetensors import load_safetensors


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
    if "model.embed_tokens.weight" in hf_tensors:
        model.token_embeddings.weight.data = hf_tensors["model.embed_tokens.weight"].copy()

    # 2. Transformer Blocks
    for i in range(model.config.num_layers):
        prefix = f"model.layers.{i}"
        block = model.blocks[i]

        # LayerNorms
        if f"{prefix}.input_layernorm.weight" in hf_tensors:
            block.attn_norm.weight.data = hf_tensors[f"{prefix}.input_layernorm.weight"].copy()
        if f"{prefix}.post_attention_layernorm.weight" in hf_tensors:
            block.mlp_norm.weight.data = hf_tensors[f"{prefix}.post_attention_layernorm.weight"].copy()

        # Attention weights
        if f"{prefix}.self_attn.q_proj.weight" in hf_tensors:
            block.attn.q_proj.weight.data = hf_tensors[f"{prefix}.self_attn.q_proj.weight"].copy()
        if f"{prefix}.self_attn.k_proj.weight" in hf_tensors:
            block.attn.k_proj.weight.data = hf_tensors[f"{prefix}.self_attn.k_proj.weight"].copy()
        if f"{prefix}.self_attn.v_proj.weight" in hf_tensors:
            block.attn.v_proj.weight.data = hf_tensors[f"{prefix}.self_attn.v_proj.weight"].copy()
        if f"{prefix}.self_attn.o_proj.weight" in hf_tensors:
            block.attn.out_proj.weight.data = hf_tensors[f"{prefix}.self_attn.o_proj.weight"].copy()

        # Attention biases (if present, e.g. Qwen2)
        if block.attn.q_proj.bias is not None and f"{prefix}.self_attn.q_proj.bias" in hf_tensors:
            block.attn.q_proj.bias.data = hf_tensors[f"{prefix}.self_attn.q_proj.bias"].copy()
        if block.attn.k_proj.bias is not None and f"{prefix}.self_attn.k_proj.bias" in hf_tensors:
            block.attn.k_proj.bias.data = hf_tensors[f"{prefix}.self_attn.k_proj.bias"].copy()
        if block.attn.v_proj.bias is not None and f"{prefix}.self_attn.v_proj.bias" in hf_tensors:
            block.attn.v_proj.bias.data = hf_tensors[f"{prefix}.self_attn.v_proj.bias"].copy()
        if block.attn.out_proj.bias is not None and f"{prefix}.self_attn.o_proj.bias" in hf_tensors:
            block.attn.out_proj.bias.data = hf_tensors[f"{prefix}.self_attn.o_proj.bias"].copy()

        # SwiGLU MLP weights
        if hasattr(block.mlp, "w_gate") and f"{prefix}.mlp.gate_proj.weight" in hf_tensors:
            block.mlp.w_gate.weight.data = hf_tensors[f"{prefix}.mlp.gate_proj.weight"].copy()
        if hasattr(block.mlp, "w_up") and f"{prefix}.mlp.up_proj.weight" in hf_tensors:
            block.mlp.w_up.weight.data = hf_tensors[f"{prefix}.mlp.up_proj.weight"].copy()
        if hasattr(block.mlp, "w_down") and f"{prefix}.mlp.down_proj.weight" in hf_tensors:
            block.mlp.w_down.weight.data = hf_tensors[f"{prefix}.mlp.down_proj.weight"].copy()

    # 3. Final Norm
    if "model.norm.weight" in hf_tensors:
        model.final_norm.weight.data = hf_tensors["model.norm.weight"].copy()

    # 4. LM Head
    if model.config.tie_weights:
        model.lm_head.weight = model.token_embeddings.weight
    elif "lm_head.weight" in hf_tensors:
        model.lm_head.weight.data = hf_tensors["lm_head.weight"].copy()

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
