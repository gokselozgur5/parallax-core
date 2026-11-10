"""
Zero-dependency HuggingFace Model Loader for ParallaxLM.

Loads pretrained HuggingFace models (LLaMA, SmolLM, Qwen2, Mistral)
from pure SafeTensors binary checkpoints without requiring `torch`
or `transformers`.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.models.llama import LlamaConfig, LlamaForCausalLM
from parallax.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    calculate_qwen2_intermediate_size,
)
from parallax.safetensors import load_safetensors, save_safetensors
from parallax.hf_tokenizer import HFTokenizer, load_hf_tokenizer
from parallax.loader_utils import _assign_bias, _assign_param


def _permute_interleaved_to_hf_weight(
    w: np.ndarray, num_heads: int, head_dim: int
) -> np.ndarray:
    """Permutes weight matrix from Parallax interleaved RoPE to HF rotate-half layout."""
    out_dim, in_dim = w.shape
    w_reshaped = w.reshape(num_heads, head_dim, in_dim)
    w_perm = np.concatenate([w_reshaped[:, 0::2, :], w_reshaped[:, 1::2, :]], axis=1)
    return w_perm.reshape(out_dim, in_dim)


def _permute_interleaved_to_hf_bias(
    b: np.ndarray, num_heads: int, head_dim: int
) -> np.ndarray:
    """Permutes bias vector from Parallax interleaved RoPE to HF rotate-half layout."""
    b_reshaped = b.reshape(num_heads, head_dim)
    b_perm = np.concatenate([b_reshaped[:, 0::2], b_reshaped[:, 1::2]], axis=1)
    return b_perm.reshape(-1)


def _unpermute_hf_to_interleaved_weight(
    w: np.ndarray, num_heads: int, head_dim: int
) -> np.ndarray:
    """Unpermutes weight matrix from HF rotate-half layout to Parallax interleaved RoPE."""
    out_dim, in_dim = w.shape
    w_reshaped = w.reshape(num_heads, head_dim, in_dim)
    half = head_dim // 2
    w_unperm = np.empty_like(w_reshaped)
    w_unperm[:, 0::2, :] = w_reshaped[:, :half, :]
    w_unperm[:, 1::2, :] = w_reshaped[:, half:, :]
    return w_unperm.reshape(out_dim, in_dim)


def _unpermute_hf_to_interleaved_bias(
    b: np.ndarray, num_heads: int, head_dim: int
) -> np.ndarray:
    """Unpermutes bias vector from HF rotate-half layout to Parallax interleaved RoPE."""
    b_reshaped = b.reshape(num_heads, head_dim)
    half = head_dim // 2
    b_unperm = np.empty_like(b_reshaped)
    b_unperm[:, 0::2] = b_reshaped[:, :half]
    b_unperm[:, 1::2] = b_reshaped[:, half:]
    return b_unperm.reshape(-1)


def hf_config_to_parallax(
    hf_config: Dict[str, Any],
    hf_tensors: Optional[Dict[str, np.ndarray]] = None,
) -> ParallaxLMConfig:
    """Translates a HuggingFace model config.json into a ParallaxLMConfig."""
    model_type = str(hf_config.get("model_type", "")).lower()
    is_supported_type = (
        model_type in ("llama", "mistral", "qwen2", "smollm")
        or model_type.startswith("qwen2")
        or model_type.startswith("smollm")
    )
    if not model_type or not is_supported_type:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Supported architectures: llama, mistral, qwen2, smollm."
        )

    if hf_config.get("rope_scaling") is not None:
        raise NotImplementedError("rope_scaling is not supported in ParallaxLM config")

    embed_dim = int(hf_config.get("hidden_size", 576))
    num_heads = int(hf_config.get("num_attention_heads", 9))

    if "head_dim" in hf_config:
        expected_head_dim = embed_dim // num_heads
        actual_head_dim = int(hf_config["head_dim"])
        if actual_head_dim != expected_head_dim:
            raise NotImplementedError(
                f"Custom head_dim ({actual_head_dim}) != hidden_size / num_heads ({expected_head_dim}) is not supported"
            )

    if "hidden_act" in hf_config:
        act = str(hf_config["hidden_act"]).lower()
        if act not in ("silu", "swish"):
            raise NotImplementedError(
                f"Unsupported hidden_act: '{act}'. Only 'silu' and 'swish' are supported for SwiGLU MLP."
            )

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
    elif model_type.startswith("qwen2"):
        intermediate_size = calculate_qwen2_intermediate_size(embed_dim)

    # Qwen2/2.5 defaults to 1e-6, others usually 1e-5
    default_eps = 1e-6 if model_type.startswith("qwen2") else 1e-5
    rms_norm_eps = float(hf_config.get("rms_norm_eps", default_eps))

    # Qwen2/2.5 defaults to 1,000,000.0, others usually 10,000.0
    default_theta = 1000000.0 if model_type.startswith("qwen2") else 10000.0
    rope_theta = float(hf_config.get("rope_theta", default_theta))

    sliding_window = hf_config.get("sliding_window")
    if sliding_window is not None:
        sliding_window = int(sliding_window)

    # Qwen2 and some architectures use attention_bias / qkv_bias
    qkv_bias = bool(
        hf_config.get("attention_bias", False)
        or hf_config.get("qkv_bias", False)
        or model_type.startswith("qwen2")
    )

    if "tie_word_embeddings" in hf_config:
        tie_weights = bool(hf_config["tie_word_embeddings"])
    else:
        if hf_tensors is not None and "lm_head.weight" in hf_tensors:
            tie_weights = False
        elif hf_tensors is not None and "lm_head.weight" not in hf_tensors:
            tie_weights = True
        else:
            tie_weights = False

    if model_type.startswith("qwen2"):
        return Qwen2Config(
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
            sliding_window=sliding_window,
        )

    if model_type == "llama":
        return LlamaConfig(
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
            sliding_window=sliding_window,
        )

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
        sliding_window=sliding_window,
    )


def load_hf_weights_into_model(
    model: ParallaxLM,
    hf_tensors: Dict[str, np.ndarray],
    strict: bool = True,
) -> ParallaxLM:
    """Maps HuggingFace tensor state dictionary directly into a ParallaxLM model."""
    expected_keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
    if not model.config.tie_weights:
        expected_keys.append("lm_head.weight")

    for i in range(model.config.num_layers):
        prefix = f"model.layers.{i}"
        expected_keys.extend([
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
        ])
        if model.config.qkv_bias:
            expected_keys.extend([
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.bias",
            ])
            # Only expect o_proj.bias if it actually exists in tensors
            if f"{prefix}.self_attn.o_proj.bias" in hf_tensors:
                expected_keys.append(f"{prefix}.self_attn.o_proj.bias")
        if model.config.use_swiglu:
            expected_keys.extend([
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ])

    missing_keys = [k for k in expected_keys if k not in hf_tensors]
    unexpected_keys = [k for k in hf_tensors if k not in expected_keys]

    model._missing_keys = missing_keys
    model._unexpected_keys = unexpected_keys

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Strict weight loading failed. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}"
        )

    # 1. Token Embeddings
    _assign_param(model.token_embeddings.weight, hf_tensors, "model.embed_tokens.weight")

    # 2. Transformer Blocks
    num_heads = model.config.num_heads
    num_kv_heads = model.config.num_kv_heads if model.config.num_kv_heads is not None else num_heads
    head_dim = model.config.embed_dim // num_heads

    for i in range(model.config.num_layers):
        prefix = f"model.layers.{i}"
        block = model.blocks[i]

        # LayerNorms
        _assign_param(block.attn_norm.weight, hf_tensors, f"{prefix}.input_layernorm.weight")
        _assign_param(block.mlp_norm.weight, hf_tensors, f"{prefix}.post_attention_layernorm.weight")

        # Attention weights
        q_key = f"{prefix}.self_attn.q_proj.weight"
        k_key = f"{prefix}.self_attn.k_proj.weight"
        if model.config.rope_interleaved and q_key in hf_tensors:
            unperm_q = _unpermute_hf_to_interleaved_weight(hf_tensors[q_key], num_heads, head_dim)
            _assign_param(block.attn.q_proj.weight, {q_key: unperm_q}, q_key)
        else:
            _assign_param(block.attn.q_proj.weight, hf_tensors, q_key)

        if model.config.rope_interleaved and k_key in hf_tensors:
            unperm_k = _unpermute_hf_to_interleaved_weight(hf_tensors[k_key], num_kv_heads, head_dim)
            _assign_param(block.attn.k_proj.weight, {k_key: unperm_k}, k_key)
        else:
            _assign_param(block.attn.k_proj.weight, hf_tensors, k_key)

        _assign_param(block.attn.v_proj.weight, hf_tensors, f"{prefix}.self_attn.v_proj.weight")
        _assign_param(block.attn.out_proj.weight, hf_tensors, f"{prefix}.self_attn.o_proj.weight")

        # Attention biases (if present, e.g. Qwen2)
        q_bias_key = f"{prefix}.self_attn.q_proj.bias"
        k_bias_key = f"{prefix}.self_attn.k_proj.bias"
        if model.config.rope_interleaved and q_bias_key in hf_tensors:
            unperm_qb = _unpermute_hf_to_interleaved_bias(hf_tensors[q_bias_key], num_heads, head_dim)
            _assign_bias(block.attn.q_proj, {q_bias_key: unperm_qb}, q_bias_key)
        else:
            _assign_bias(block.attn.q_proj, hf_tensors, q_bias_key)

        if model.config.rope_interleaved and k_bias_key in hf_tensors:
            unperm_kb = _unpermute_hf_to_interleaved_bias(hf_tensors[k_bias_key], num_kv_heads, head_dim)
            _assign_bias(block.attn.k_proj, {k_bias_key: unperm_kb}, k_bias_key)
        else:
            _assign_bias(block.attn.k_proj, hf_tensors, k_bias_key)

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


def load_hf_pretrained(
    model_dir: str,
    return_tokenizer: bool = False,
    strict: bool = True,
) -> Union[ParallaxLM, Tuple[ParallaxLM, Optional[HFTokenizer]]]:
    """Loads a pretrained HuggingFace model directory into a ready ParallaxLM instance, optionally returning its tokenizer."""
    config_file = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    # Check for single-file or sharded safetensors
    weights_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(weights_file):
        weight_files = [weights_file]
    else:
        sf_candidates = sorted(
            f for f in os.listdir(model_dir) if f.endswith(".safetensors")
        )
        if not sf_candidates:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
        weight_files = [os.path.join(model_dir, f) for f in sf_candidates]

    tensors: Dict[str, np.ndarray] = {}
    for wf in weight_files:
        tensors.update(load_safetensors(wf, cast_float64=True))
    p_config = hf_config_to_parallax(hf_config, hf_tensors=tensors)
    if isinstance(p_config, Qwen2Config) or (isinstance(hf_config.get("model_type"), str) and hf_config.get("model_type", "").startswith("qwen2")):
        model = Qwen2ForCausalLM(p_config)
    elif isinstance(p_config, LlamaConfig) or hf_config.get("model_type") == "llama":
        model = LlamaForCausalLM(p_config)
    else:
        model = ParallaxLM(p_config)
    load_hf_weights_into_model(model, tensors, strict=strict)

    if return_tokenizer:
        tok_file = os.path.join(model_dir, "tokenizer.json")
        tokenizer = load_hf_tokenizer(tok_file) if os.path.exists(tok_file) else None
        return model, tokenizer

    return model


def save_hf_pretrained(
    model: ParallaxLM,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    model_type: str = "llama",
) -> None:
    """Exports a ParallaxLM model into standard HuggingFace format (config.json + model.safetensors)."""
    cfg = model.config

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

    if not cfg.use_swiglu:
        raise NotImplementedError(
            "Saving GELU-MLP models to HuggingFace format is not supported (HuggingFace LLaMA/Qwen2 requires SwiGLU gate/up/down projections)."
        )

    os.makedirs(output_dir, exist_ok=True)

    if cfg.intermediate_size is not None:
        intermediate_size = cfg.intermediate_size
    elif cfg.use_swiglu:
        inter = int(2 * (4 * cfg.embed_dim) / 3)
        intermediate_size = 64 * ((inter + 64 - 1) // 64)
    else:
        intermediate_size = 4 * cfg.embed_dim
    num_kv_heads = cfg.num_kv_heads if cfg.num_kv_heads is not None else cfg.num_heads

    # 1. Write config.json
    hf_config = {
        "architectures": ["LlamaForCausalLM"] if model_type == "llama" else ["Qwen2ForCausalLM"],
        "model_type": model_type,
        "hidden_size": int(cfg.embed_dim),
        "num_attention_heads": int(cfg.num_heads),
        "num_key_value_heads": int(num_kv_heads),
        "num_hidden_layers": int(cfg.num_layers),
        "vocab_size": int(cfg.vocab_size),
        "max_position_embeddings": int(cfg.max_seq_len),
        "intermediate_size": int(intermediate_size),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "rope_theta": float(cfg.rope_theta),
        "attention_bias": bool(cfg.qkv_bias),
        "tie_word_embeddings": bool(cfg.tie_weights),
        "torch_dtype": "float32",
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # 2. Collect model.safetensors
    tensors: Dict[str, np.ndarray] = {
        "model.embed_tokens.weight": model.token_embeddings.weight.data.astype(np.float32),
        "model.norm.weight": model.final_norm.weight.data.astype(np.float32),
    }

    if not cfg.tie_weights:
        tensors["lm_head.weight"] = model.lm_head.weight.data.astype(np.float32)

    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}"
        block = model.blocks[i]

        tensors[f"{prefix}.input_layernorm.weight"] = block.attn_norm.weight.data.astype(np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = block.mlp_norm.weight.data.astype(np.float32)

        q_w = block.attn.q_proj.weight.data.astype(np.float32)
        k_w = block.attn.k_proj.weight.data.astype(np.float32)
        head_dim = cfg.embed_dim // cfg.num_heads
        if cfg.rope_interleaved:
            q_w = _permute_interleaved_to_hf_weight(q_w, cfg.num_heads, head_dim)
            k_w = _permute_interleaved_to_hf_weight(k_w, num_kv_heads, head_dim)

        tensors[f"{prefix}.self_attn.q_proj.weight"] = q_w
        tensors[f"{prefix}.self_attn.k_proj.weight"] = k_w
        tensors[f"{prefix}.self_attn.v_proj.weight"] = block.attn.v_proj.weight.data.astype(np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = block.attn.out_proj.weight.data.astype(np.float32)

        if block.attn.q_proj.bias is not None:
            q_b = block.attn.q_proj.bias.data.astype(np.float32)
            if cfg.rope_interleaved:
                q_b = _permute_interleaved_to_hf_bias(q_b, cfg.num_heads, head_dim)
            tensors[f"{prefix}.self_attn.q_proj.bias"] = q_b
        elif cfg.qkv_bias:
            tensors[f"{prefix}.self_attn.q_proj.bias"] = np.zeros(block.attn.q_proj.out_features, dtype=np.float32)

        if block.attn.k_proj.bias is not None:
            k_b = block.attn.k_proj.bias.data.astype(np.float32)
            if cfg.rope_interleaved:
                k_b = _permute_interleaved_to_hf_bias(k_b, num_kv_heads, head_dim)
            tensors[f"{prefix}.self_attn.k_proj.bias"] = k_b
        elif cfg.qkv_bias:
            tensors[f"{prefix}.self_attn.k_proj.bias"] = np.zeros(block.attn.k_proj.out_features, dtype=np.float32)

        if block.attn.v_proj.bias is not None:
            tensors[f"{prefix}.self_attn.v_proj.bias"] = block.attn.v_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"{prefix}.self_attn.v_proj.bias"] = np.zeros(block.attn.v_proj.out_features, dtype=np.float32)

        if block.attn.out_proj.bias is not None:
            tensors[f"{prefix}.self_attn.o_proj.bias"] = block.attn.out_proj.bias.data.astype(np.float32)
        elif cfg.qkv_bias:
            tensors[f"{prefix}.self_attn.o_proj.bias"] = np.zeros(block.attn.out_proj.out_features, dtype=np.float32)

        if hasattr(block.mlp, "w_gate"):
            tensors[f"{prefix}.mlp.gate_proj.weight"] = block.mlp.w_gate.weight.data.astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = block.mlp.w_up.weight.data.astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = block.mlp.w_down.weight.data.astype(np.float32)
        elif hasattr(block.mlp, "fc1"):
            tensors[f"{prefix}.mlp.gate_proj.weight"] = block.mlp.fc1.weight.data.astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = block.mlp.fc1.weight.data.astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = block.mlp.fc2.weight.data.astype(np.float32)

    save_safetensors(os.path.join(output_dir, "model.safetensors"), tensors)

    # 3. Optional tokenizer serialization
    if tokenizer is not None:
        if hasattr(tokenizer, "vocab"):
            added_tokens = []
            if hasattr(tokenizer, "added_tokens"):
                for content, tid in tokenizer.added_tokens.items():
                    added_tokens.append({"id": tid, "content": content, "special": True})

            merges_list = []
            if hasattr(tokenizer, "merges") and isinstance(tokenizer.merges, dict):
                # Check whether keys are tuple of str or tuple of int
                for pair in tokenizer.merges.keys():
                    if isinstance(pair[0], str):
                        merges_list.append(f"{pair[0]} {pair[1]}")
                    elif hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
                        # Convert integer tuple to tokens
                        inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
                        t0 = inv_vocab.get(pair[0], "")
                        t1 = inv_vocab.get(pair[1], "")
                        if t0 and t1:
                            merges_list.append(f"{t0} {t1}")

            tok_data = {
                "version": "1.0",
                "added_tokens": added_tokens,
                "model": {
                    "type": "BPE",
                    "vocab": tokenizer.vocab,
                    "merges": merges_list,
                },
            }
            with open(os.path.join(output_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(tok_data, f, indent=2)
