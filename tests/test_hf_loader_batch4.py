"""
Tests for HuggingFace / GGUF loader validation and exporter restrictions.
Issues #225, #227, #228, #230, #232.
"""

import os
import json
import tempfile
import numpy as np
import pytest

from parallax.hf_loader import hf_config_to_parallax, save_hf_pretrained
from parallax.gguf_loader import gguf_metadata_to_parallax, save_parallax_to_gguf
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_hf_config_unsupported_model_type():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        hf_config_to_parallax({"model_type": "falcon", "hidden_size": 128})

    with pytest.raises(ValueError, match="Unsupported model_type"):
        hf_config_to_parallax({"hidden_size": 128})


def test_gguf_metadata_unsupported_arch():
    with pytest.raises(ValueError, match="Unsupported GGUF architecture"):
        gguf_metadata_to_parallax({"general.architecture": "gpt2"})


def test_hf_config_rope_scaling_rejected():
    with pytest.raises(NotImplementedError, match="rope_scaling is not supported"):
        hf_config_to_parallax({
            "model_type": "llama",
            "hidden_size": 128,
            "rope_scaling": {"type": "linear", "factor": 2.0},
        })


def test_hf_config_custom_head_dim_mismatch():
    # embed_dim=128, num_heads=4 -> expected head_dim = 32
    with pytest.raises(NotImplementedError, match="Custom head_dim"):
        hf_config_to_parallax({
            "model_type": "llama",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "head_dim": 64,
        })


def test_hf_config_hidden_act_validation():
    with pytest.raises(NotImplementedError, match="Unsupported hidden_act"):
        hf_config_to_parallax({
            "model_type": "llama",
            "hidden_size": 128,
            "hidden_act": "relu",
        })

    with pytest.raises(NotImplementedError, match="Unsupported hidden_act"):
        hf_config_to_parallax({
            "model_type": "llama",
            "hidden_size": 128,
            "hidden_act": "gelu",
        })

    cfg_silu = hf_config_to_parallax({
        "model_type": "llama",
        "hidden_size": 128,
        "hidden_act": "silu",
    })
    assert cfg_silu.use_swiglu is True

    cfg_swish = hf_config_to_parallax({
        "model_type": "llama",
        "hidden_size": 128,
        "hidden_act": "swish",
    })
    assert cfg_swish.use_swiglu is True


def test_hf_config_tie_word_embeddings_defaults():
    # If tie_word_embeddings not provided and tensors provided without lm_head.weight: True
    cfg_tied = hf_config_to_parallax(
        {"model_type": "llama", "hidden_size": 128},
        hf_tensors={"model.embed_tokens.weight": np.zeros((10, 128))},
    )
    assert cfg_tied.tie_weights is True

    # If lm_head.weight is in tensors: False
    cfg_untied = hf_config_to_parallax(
        {"model_type": "llama", "hidden_size": 128},
        hf_tensors={"lm_head.weight": np.zeros((10, 128))},
    )
    assert cfg_untied.tie_weights is False


def test_hf_exporter_intermediate_size_formula():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=64,
        use_swiglu=True,
        intermediate_size=None,
    )
    model = ParallaxLM(cfg)
    inter = int(2 * (4 * cfg.embed_dim) / 3)
    expected_inter = 64 * ((inter + 64 - 1) // 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_pretrained(model, tmpdir, model_type="llama")
        with open(os.path.join(tmpdir, "config.json"), "r", encoding="utf-8") as f:
            saved_cfg = json.load(f)
        assert saved_cfg["intermediate_size"] == expected_inter


def test_hf_exporter_refuses_non_swiglu():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(NotImplementedError, match="Saving GELU-MLP models"):
            save_hf_pretrained(model, tmpdir)


def test_exporters_refuse_moe_model():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=64,
        use_swiglu=True,
        num_experts=4,
        top_k_experts=2,
    )
    model = ParallaxLM(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(NotImplementedError, match="MoE model export is not supported"):
            save_hf_pretrained(model, os.path.join(tmpdir, "hf"))

        with pytest.raises(NotImplementedError, match="MoE model export is not supported"):
            save_parallax_to_gguf(model, os.path.join(tmpdir, "model.gguf"))
