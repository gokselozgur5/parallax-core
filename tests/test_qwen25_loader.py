"""
Tests for HuggingFace Qwen2 / Qwen2.5 model family loader, configuration, and inference pipeline.
"""

import json
import os
import tempfile
import numpy as np

from parallax.models.qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    calculate_qwen2_intermediate_size,
)
from parallax.hf_loader import (
    hf_config_to_parallax,
    load_hf_pretrained,
)
from parallax.safetensors import save_safetensors


def test_qwen2_intermediate_size_formula():
    """Verify SwiGLU intermediate size calculation formula."""
    # Hidden size = 1024 -> raw = 2 * (4 * 1024) / 3 = 2730.66 -> rounded to multiple of 256 is 2816
    assert calculate_qwen2_intermediate_size(1024, multiple_of=256) == 2816
    assert calculate_qwen2_intermediate_size(512, multiple_of=64) == 1408


def test_qwen25_coder_preset_configs():
    """Test predefined Qwen2.5 architectural preset configurations."""
    # 1. Qwen2.5-0.5B preset
    cfg_05b = Qwen2Config.qwen2_5_0_5b()
    assert cfg_05b.vocab_size == 151936
    assert cfg_05b.embed_dim == 896
    assert cfg_05b.num_heads == 14
    assert cfg_05b.num_kv_heads == 2
    assert cfg_05b.num_layers == 24
    assert cfg_05b.intermediate_size == 4864
    assert cfg_05b.rms_norm_eps == 1e-6
    assert cfg_05b.rope_theta == 1000000.0
    assert cfg_05b.qkv_bias is True
    assert cfg_05b.tie_weights is True
    assert cfg_05b.sliding_window == 32768
    assert cfg_05b.rope_interleaved is False

    # 2. Qwen2.5-Coder-1.5B preset
    cfg_15b = Qwen2Config.qwen2_5_coder_1_5b()
    assert cfg_15b.vocab_size == 151936
    assert cfg_15b.embed_dim == 1536
    assert cfg_15b.num_heads == 12
    assert cfg_15b.num_kv_heads == 2
    assert cfg_15b.num_layers == 28
    assert cfg_15b.intermediate_size == 8960
    assert cfg_15b.rms_norm_eps == 1e-6
    assert cfg_15b.rope_theta == 1000000.0
    assert cfg_15b.qkv_bias is True
    assert cfg_15b.tie_weights is True
    assert cfg_15b.sliding_window == 32768

    # 3. Qwen2.5-Coder-7B preset
    cfg_7b = Qwen2Config.qwen2_5_coder_7b()
    assert cfg_7b.vocab_size == 152064
    assert cfg_7b.embed_dim == 3584
    assert cfg_7b.num_heads == 28
    assert cfg_7b.num_kv_heads == 4
    assert cfg_7b.num_layers == 28
    assert cfg_7b.intermediate_size == 18944
    assert cfg_7b.rms_norm_eps == 1e-6
    assert cfg_7b.rope_theta == 1000000.0
    assert cfg_7b.qkv_bias is True
    assert cfg_7b.tie_weights is False
    assert cfg_7b.sliding_window == 131072


def test_hf_config_to_parallax_qwen25_translation():
    """Verify HF config.json mapping to Qwen2Config with Q/K/V bias, eps=1e-6, rope_theta=1M."""
    hf_cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": 1536,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "intermediate_size": 8960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": True,
    }
    p_cfg = hf_config_to_parallax(hf_cfg)
    assert isinstance(p_cfg, Qwen2Config)
    assert p_cfg.embed_dim == 1536
    assert p_cfg.num_heads == 12
    assert p_cfg.num_kv_heads == 2
    assert p_cfg.num_layers == 28
    assert p_cfg.vocab_size == 151936
    assert p_cfg.intermediate_size == 8960
    assert p_cfg.rms_norm_eps == 1e-6
    assert p_cfg.rope_theta == 1000000.0
    assert p_cfg.qkv_bias is True
    assert p_cfg.tie_weights is True
    assert p_cfg.sliding_window == 32768
    assert p_cfg.rope_interleaved is False


def test_qwen25_synthetic_loader_and_forward_pass():
    """Build synthetic Qwen2.5 safetensors checkpoint and verify forward pass end-to-end."""
    np.random.seed(42)

    V = 1000     # vocab size
    D = 64       # embed_dim
    H = 4        # num_heads (head_dim = 16)
    KV_H = 2     # num_kv_heads (GQA 2:1)
    L = 2        # num_layers
    I = 128      # intermediate_size (SwiGLU)
    ROPE_THETA = 1000000.0
    RMS_EPS = 1e-6
    SLIDING_WINDOW = 32

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. HuggingFace config.json without explicit attention_bias key (Qwen standard)
        hf_cfg = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": D,
            "num_attention_heads": H,
            "num_key_value_heads": KV_H,
            "num_hidden_layers": L,
            "vocab_size": V,
            "max_position_embeddings": 256,
            "intermediate_size": I,
            "rms_norm_eps": RMS_EPS,
            "rope_theta": ROPE_THETA,
            "sliding_window": SLIDING_WINDOW,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f, indent=2)

        # 2. Synthetic SafeTensors weights matching Qwen2.5 state_dict
        head_dim = D // H
        kv_dim = KV_H * head_dim

        tensors = {
            "model.embed_tokens.weight": np.random.randn(V, D).astype(np.float32) * 0.02,
            "model.norm.weight": np.ones((D,), dtype=np.float32),
            "lm_head.weight": np.random.randn(V, D).astype(np.float32) * 0.02,
        }

        for i in range(L):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = np.ones((D,), dtype=np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones((D,), dtype=np.float32)

            # Attention weights and Q/K/V/O biases
            tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(D, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.q_proj.bias"] = np.random.randn(D).astype(np.float32) * 0.01
            tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(kv_dim, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.k_proj.bias"] = np.random.randn(kv_dim).astype(np.float32) * 0.01
            tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(kv_dim, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.v_proj.bias"] = np.random.randn(kv_dim).astype(np.float32) * 0.01
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(D, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.o_proj.bias"] = np.zeros(D, dtype=np.float32)

            # SwiGLU MLP: gate_proj (I, D), up_proj (I, D), down_proj (D, I)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(I, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(I, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(D, I).astype(np.float32) * 0.02

        weights_path = os.path.join(tmpdir, "model.safetensors")
        save_safetensors(weights_path, tensors)

        # 3. Load model via load_hf_pretrained
        model = load_hf_pretrained(tmpdir, strict=True)
        assert isinstance(model, Qwen2ForCausalLM)
        assert isinstance(model.config, Qwen2Config)
        assert model.config.vocab_size == V
        assert model.config.rope_theta == ROPE_THETA
        assert model.config.rms_norm_eps == RMS_EPS
        assert model.config.num_kv_heads == KV_H
        assert model.config.embed_dim == D
        assert model.config.qkv_bias is True
        assert model.config.sliding_window == SLIDING_WINDOW

        # 4. Verify parameter bindings and biases
        np.testing.assert_allclose(
            model.token_embeddings.weight.data, tensors["model.embed_tokens.weight"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.lm_head.weight.data, tensors["lm_head.weight"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.q_proj.bias.data, tensors["model.layers.0.self_attn.q_proj.bias"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.k_proj.bias.data, tensors["model.layers.0.self_attn.k_proj.bias"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.v_proj.bias.data, tensors["model.layers.0.self_attn.v_proj.bias"], rtol=1e-5
        )

        # 5. Forward pass / inference check
        batch_size = 2
        seq_len = 8
        input_ids = np.random.randint(0, V, size=(batch_size, seq_len), dtype=np.int64)

        logits = model(input_ids)
        assert logits.shape == (batch_size, seq_len, V), f"Expected {(batch_size, seq_len, V)}, got {logits.shape}"
        assert not np.isnan(logits.data).any(), "Logits contain NaN values"
        assert not np.isinf(logits.data).any(), "Logits contain Inf values"

        # 6. KV-cache autoregressive forward step
        kv_caches = [
            (
                np.zeros((batch_size, KV_H, 0, head_dim), dtype=np.float64),
                np.zeros((batch_size, KV_H, 0, head_dim), dtype=np.float64),
            )
            for _ in range(L)
        ]
        step_logits, updated_caches = model(input_ids, kv_caches=kv_caches)
        assert step_logits.shape == (batch_size, seq_len, V)
        assert len(updated_caches) == L
        assert updated_caches[0][0].shape == (batch_size, KV_H, seq_len, head_dim)
