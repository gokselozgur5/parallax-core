import os
import tempfile
import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.hf_loader import (
    load_hf_weights_into_model,
    load_hf_pretrained,
    save_hf_pretrained,
)
from parallax.gguf_loader import (
    load_gguf_weights_into_model,
    load_gguf_pretrained,
    save_parallax_to_gguf,
)


def _create_dummy_model(rope_interleaved: bool = True, qkv_bias: bool = False, tie_weights: bool = True) -> ParallaxLM:
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        use_rope=True,
        rope_interleaved=rope_interleaved,
        use_swiglu=True,
        intermediate_size=32,
        rms_norm_eps=1e-5,
        tie_weights=tie_weights,
        qkv_bias=qkv_bias,
    )
    return ParallaxLM(cfg)


def test_hf_loader_strict_missing_keys():
    model = _create_dummy_model()
    tensors = {
        "model.embed_tokens.weight": np.zeros_like(model.token_embeddings.weight.data),
    }
    with pytest.raises(RuntimeError, match="Strict weight loading failed. Missing keys:"):
        load_hf_weights_into_model(model, tensors, strict=True)

    # strict=False should succeed and set _missing_keys
    load_hf_weights_into_model(model, tensors, strict=False)
    assert len(model._missing_keys) > 0
    assert "model.norm.weight" in model._missing_keys


def test_hf_loader_strict_unexpected_keys():
    model = _create_dummy_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_pretrained(model, tmpdir)
        # Load weights and add an unexpected key
        from parallax.safetensors import load_safetensors
        tensors = load_safetensors(os.path.join(tmpdir, "model.safetensors"), cast_float64=True)
        tensors["unexpected.extra.weight"] = np.zeros((4, 4), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Unexpected keys:"):
            load_hf_weights_into_model(model, tensors, strict=True)

        load_hf_weights_into_model(model, tensors, strict=False)
        assert "unexpected.extra.weight" in model._unexpected_keys


def test_gguf_loader_strict_missing_and_unexpected_keys():
    model = _create_dummy_model()
    tensors = {
        "token_embd.weight": np.zeros_like(model.token_embeddings.weight.data),
    }
    with pytest.raises(RuntimeError, match="Strict GGUF weight loading failed. Missing:"):
        load_gguf_weights_into_model(model, tensors, strict=True)

    load_gguf_weights_into_model(model, tensors, strict=False)
    assert "output_norm.weight" in model._missing_keys

    with tempfile.TemporaryDirectory() as tmpdir:
        gguf_path = os.path.join(tmpdir, "model.gguf")
        save_parallax_to_gguf(model, gguf_path)
        from parallax.gguf import load_gguf
        gf = load_gguf(gguf_path, cast_float64=True)
        tensors = dict(gf.tensors)
        tensors["extra.fake.tensor"] = np.zeros((2,), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Unexpected:"):
            load_gguf_weights_into_model(model, tensors, strict=True)


def test_hf_roundtrip_rope_interleaved_true():
    np.random.seed(42)
    model = _create_dummy_model(rope_interleaved=True, qkv_bias=True, tie_weights=False)
    # Randomize weights
    for p in model.parameters():
        p.data = np.random.randn(*p.data.shape).astype(np.float32)

    idx = np.array([[1, 5, 12, 8, 3]], dtype=np.int64)
    logits_before = model(idx).data

    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_pretrained(model, tmpdir)
        # Re-load model with exact same config (rope_interleaved=True)
        model_loaded = ParallaxLM(model.config)
        from parallax.safetensors import load_safetensors
        tensors = load_safetensors(os.path.join(tmpdir, "model.safetensors"), cast_float64=True)
        load_hf_weights_into_model(model_loaded, tensors, strict=True)

        logits_after = model_loaded(idx).data
        diff = np.max(np.abs(logits_before - logits_after))
        assert diff < 1e-5, f"Logits diff {diff} exceeded 1e-5 for rope_interleaved=True"


def test_hf_roundtrip_rope_interleaved_false():
    np.random.seed(42)
    model = _create_dummy_model(rope_interleaved=False, qkv_bias=True, tie_weights=False)
    for p in model.parameters():
        p.data = np.random.randn(*p.data.shape).astype(np.float32)

    idx = np.array([[1, 5, 12, 8, 3]], dtype=np.int64)
    logits_before = model(idx).data

    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_pretrained(model, tmpdir)
        loaded = load_hf_pretrained(tmpdir, strict=True)
        logits_after = loaded(idx).data
        diff = np.max(np.abs(logits_before - logits_after))
        assert diff < 1e-5, f"Logits diff {diff} exceeded 1e-5 for rope_interleaved=False"
