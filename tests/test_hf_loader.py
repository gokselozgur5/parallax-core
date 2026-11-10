"""
End-to-end tests for zero-dependency HuggingFace model loader.
"""

import json
import os
import tempfile
import numpy as np
import pytest

from parallax.generator import generate
from parallax.hf_loader import (
    hf_config_to_parallax,
    load_hf_pretrained,
    load_hf_weights_into_model,
)
from parallax.safetensors import save_safetensors


def test_hf_config_to_parallax_translation():
    hf_cfg = {
        "model_type": "llama",
        "hidden_size": 576,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "num_hidden_layers": 30,
        "vocab_size": 49152,
        "max_position_embeddings": 2048,
        "intermediate_size": 1536,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0,
        "attention_bias": True,
        "tie_word_embeddings": True,
    }
    p_cfg = hf_config_to_parallax(hf_cfg)
    assert p_cfg.embed_dim == 576
    assert p_cfg.num_heads == 9
    assert p_cfg.num_kv_heads == 3
    assert p_cfg.num_layers == 30
    assert p_cfg.vocab_size == 49152
    assert p_cfg.max_seq_len == 2048
    assert p_cfg.intermediate_size == 1536
    assert p_cfg.rms_norm_eps == 1e-5
    assert p_cfg.rope_theta == 100000.0
    assert p_cfg.qkv_bias is True
    assert p_cfg.tie_weights is True
    assert p_cfg.use_swiglu is True
    assert p_cfg.rope_interleaved is False


def test_load_hf_pretrained_and_inference_e2e():
    np.random.seed(42)
    V, D, H, L, I = 32, 16, 2, 2, 32

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write HuggingFace config.json
        hf_cfg = {
            "model_type": "llama",
            "hidden_size": D,
            "num_attention_heads": H,
            "num_key_value_heads": H,
            "num_hidden_layers": L,
            "vocab_size": V,
            "max_position_embeddings": 64,
            "intermediate_size": I,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_bias": True,
            "tie_word_embeddings": True,
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f)

        # 2. Prepare synthetic weights dictionary
        tensors = {
            "model.embed_tokens.weight": np.random.randn(V, D).astype(np.float32),
            "model.norm.weight": np.ones((D,), dtype=np.float32),
        }
        for i in range(L):
            prefix = f"model.layers.{i}"
            tensors[f"{prefix}.input_layernorm.weight"] = np.ones((D,), dtype=np.float32)
            tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones((D,), dtype=np.float32)

            tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(D, D).astype(np.float32)
            tensors[f"{prefix}.self_attn.q_proj.bias"] = np.zeros((D,), dtype=np.float32)
            tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(D, D).astype(np.float32)
            tensors[f"{prefix}.self_attn.k_proj.bias"] = np.zeros((D,), dtype=np.float32)
            tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(D, D).astype(np.float32)
            tensors[f"{prefix}.self_attn.v_proj.bias"] = np.zeros((D,), dtype=np.float32)
            tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(D, D).astype(np.float32)
            tensors[f"{prefix}.self_attn.o_proj.bias"] = np.zeros((D,), dtype=np.float32)

            tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(I, D).astype(np.float32)
            tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(I, D).astype(np.float32)
            tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(D, I).astype(np.float32)

        # 3. Save to safetensors
        weights_path = os.path.join(tmpdir, "model.safetensors")
        save_safetensors(weights_path, tensors)

        # 4. Load full pretrained model using zero-dependency loader
        model = load_hf_pretrained(tmpdir)

        # Verify weights were correctly mapped into ParallaxLM
        np.testing.assert_allclose(
            model.token_embeddings.weight.data,
            tensors["model.embed_tokens.weight"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.q_proj.weight.data,
            tensors["model.layers.0.self_attn.q_proj.weight"],
            rtol=1e-5,
        )
        assert model.lm_head.weight is model.token_embeddings.weight

        # 5. Verify forward pass
        idx = np.array([[1, 5, 12, 8]], dtype=np.int64)
        logits = model(idx)
        assert logits.shape == (1, 4, V)
        assert not np.isnan(logits.data).any()

        # 6. Verify autoregressive generation with KV cache
        prompt = [1, 5]
        generated = generate(model, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
        assert len(generated) == len(prompt) + 4
        assert all(0 <= tok < V for tok in generated)


def test_load_hf_untied_lm_head():
    np.random.seed(42)
    V, D, H, L, I = 20, 8, 2, 1, 16

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_cfg = {
            "model_type": "llama",
            "hidden_size": D,
            "num_attention_heads": H,
            "num_hidden_layers": L,
            "vocab_size": V,
            "intermediate_size": I,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f)

        embed_w = np.random.randn(V, D).astype(np.float32)
        head_w = np.random.randn(V, D).astype(np.float32)

        tensors = {
            "model.embed_tokens.weight": embed_w,
            "model.norm.weight": np.ones((D,), dtype=np.float32),
            "lm_head.weight": head_w,
            "model.layers.0.input_layernorm.weight": np.ones((D,), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones((D,), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(I, D).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(I, D).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(D, I).astype(np.float32),
        }
        save_safetensors(os.path.join(tmpdir, "model.safetensors"), tensors)

        model = load_hf_pretrained(tmpdir)
        assert model.lm_head.weight is not model.token_embeddings.weight
        np.testing.assert_allclose(model.lm_head.weight.data, head_w, rtol=1e-5)


def test_qwen2_default_qkv_bias_and_sharded_safetensors():
    np.random.seed(7)
    V, D, H, L, I = 16, 8, 2, 1, 16

    with tempfile.TemporaryDirectory() as tmpdir:
        # Qwen2 config omits attention_bias in HF config.json
        hf_cfg = {
            "model_type": "qwen2",
            "hidden_size": D,
            "num_attention_heads": H,
            "num_key_value_heads": H,
            "num_hidden_layers": L,
            "vocab_size": V,
            "intermediate_size": I,
            "tie_word_embeddings": True,
        }
        p_cfg = hf_config_to_parallax(hf_cfg)
        assert p_cfg.qkv_bias is True

        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(hf_cfg, f)

        q_bias = np.full((D,), 0.25, dtype=np.float32)
        o_bias = np.full((D,), -0.5, dtype=np.float32)

        shard1 = {
            "model.embed_tokens.weight": np.random.randn(V, D).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.ones((D,), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.q_proj.bias": q_bias,
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.k_proj.bias": np.zeros((D,), dtype=np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.v_proj.bias": np.zeros((D,), dtype=np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(D, D).astype(np.float32),
            "model.layers.0.self_attn.o_proj.bias": o_bias,
        }
        shard2 = {
            "model.norm.weight": np.ones((D,), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones((D,), dtype=np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(I, D).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(I, D).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(D, I).astype(np.float32),
        }
        save_safetensors(os.path.join(tmpdir, "model-00001-of-00002.safetensors"), shard1)
        save_safetensors(os.path.join(tmpdir, "model-00002-of-00002.safetensors"), shard2)

        model = load_hf_pretrained(tmpdir)
        np.testing.assert_allclose(model.blocks[0].attn.q_proj.bias.data, q_bias, rtol=1e-5)
        assert model.blocks[0].attn.out_proj.bias is not None
        np.testing.assert_allclose(model.blocks[0].attn.out_proj.bias.data, o_bias, rtol=1e-5)
        np.testing.assert_allclose(
            model.blocks[0].mlp.w_down.weight.data,
            shard2["model.layers.0.mlp.down_proj.weight"],
            rtol=1e-5,
        )

        # Verify shape mismatch raises ValueError
        bad_tensors = {"model.embed_tokens.weight": np.zeros((V + 1, D), dtype=np.float64)}
        with pytest.raises(ValueError, match="Shape mismatch"):
            load_hf_weights_into_model(model, bad_tensors, strict=False)
