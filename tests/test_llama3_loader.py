"""
Tests for HuggingFace LLaMA-3 model architecture, loader, and inference pipeline.
"""

import json
import os
import tempfile
import numpy as np

from parallax.models.llama import LlamaConfig, LlamaForCausalLM
from parallax.hf_loader import (
    hf_config_to_parallax,
    load_hf_pretrained,
)
from parallax.safetensors import save_safetensors


def test_llama3_8b_default_config():
    """Verify default architectural parameters for Meta-Llama-3-8B."""
    cfg = LlamaConfig.llama_3_8b()
    assert cfg.vocab_size == 128256
    assert cfg.embed_dim == 4096
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 8  # Grouped-Query Attention (GQA) 4:1
    assert cfg.num_layers == 32
    assert cfg.max_seq_len == 8192
    assert cfg.intermediate_size == 14336
    assert cfg.rope_theta == 500000.0  # LLaMA-3 RoPE base frequency
    assert cfg.rms_norm_eps == 1e-5
    assert cfg.use_swiglu is True
    assert cfg.qkv_bias is False
    assert cfg.tie_weights is False
    assert cfg.rope_interleaved is False


def test_hf_config_to_parallax_llama3_mapping():
    """Test translating LLaMA-3 style HuggingFace config.json to LlamaConfig."""
    hf_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "intermediate_size": 14336,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    p_cfg = hf_config_to_parallax(hf_cfg)
    assert isinstance(p_cfg, LlamaConfig)
    assert p_cfg.vocab_size == 128256
    assert p_cfg.embed_dim == 4096
    assert p_cfg.num_heads == 32
    assert p_cfg.num_kv_heads == 8
    assert p_cfg.num_layers == 32
    assert p_cfg.intermediate_size == 14336
    assert p_cfg.rope_theta == 500000.0
    assert p_cfg.rms_norm_eps == 1e-5
    assert p_cfg.use_swiglu is True
    assert p_cfg.tie_weights is False
    assert p_cfg.rope_interleaved is False


def test_synthetic_llama3_loader_and_inference():
    """Build synthetic Llama-3 safetensors checkpoint and verify pipeline end-to-end.

    Validates:
    1. Zero-dependency safetensors serialization and deserialization.
    2. HF config parsing into LlamaConfig.
    3. Proper parameter mapping into LlamaForCausalLM.
    4. GQA attention forward pass with RoPE theta=500000.
    5. Correct output logits shape [B, T, vocab_size] with vocab_size=128256.
    """
    np.random.seed(42)

    # Scaling down hidden dims for fast unit testing while keeping Llama-3 architecture intact
    V = 128256  # Standard LLaMA-3 vocab size
    D = 64      # embed_dim
    H = 4       # num_heads
    KV_H = 2    # num_kv_heads (GQA ratio 2:1)
    L = 2       # num_layers
    I = 128     # intermediate_size (SwiGLU)
    ROPE_THETA = 500000.0
    RMS_EPS = 1e-5

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. HuggingFace config.json
        hf_cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": D,
            "num_attention_heads": H,
            "num_key_value_heads": KV_H,
            "num_hidden_layers": L,
            "vocab_size": V,
            "max_position_embeddings": 256,
            "intermediate_size": I,
            "rms_norm_eps": RMS_EPS,
            "rope_theta": ROPE_THETA,
            "attention_bias": False,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f, indent=2)

        # 2. Synthetic SafeTensors weights matching LLaMA-3 state_dict
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

            # Attention projections: Q (D, D), K (kv_dim, D), V (kv_dim, D), O (D, D)
            tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(D, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(kv_dim, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(kv_dim, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(D, D).astype(np.float32) * 0.02

            # SwiGLU MLP: gate_proj (I, D), up_proj (I, D), down_proj (D, I)
            tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(I, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(I, D).astype(np.float32) * 0.02
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(D, I).astype(np.float32) * 0.02

        weights_path = os.path.join(tmpdir, "model.safetensors")
        save_safetensors(weights_path, tensors)

        # 3. Load model via load_hf_pretrained
        model = load_hf_pretrained(tmpdir, strict=True)
        assert isinstance(model, LlamaForCausalLM)
        assert model.config.vocab_size == V
        assert model.config.rope_theta == ROPE_THETA
        assert model.config.num_kv_heads == KV_H
        assert model.config.embed_dim == D

        # 4. Verify weight binding
        np.testing.assert_allclose(
            model.token_embeddings.weight.data, tensors["model.embed_tokens.weight"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.lm_head.weight.data, tensors["lm_head.weight"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.q_proj.weight.data, tensors["model.layers.0.self_attn.q_proj.weight"], rtol=1e-5
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.k_proj.weight.data, tensors["model.layers.0.self_attn.k_proj.weight"], rtol=1e-5
        )

        # 5. Forward pass / inference check
        batch_size = 2
        seq_len = 5
        input_ids = np.random.randint(0, V, size=(batch_size, seq_len), dtype=np.int64)

        logits = model(input_ids)

        # Verify output shape [B, T, vocab_size]
        assert logits.shape == (batch_size, seq_len, V), f"Expected shape {(batch_size, seq_len, V)}, got {logits.shape}"
        assert not np.isnan(logits.data).any(), "Logits contain NaN values"
        assert not np.isinf(logits.data).any(), "Logits contain Inf values"
