import os
import tempfile
import numpy as np

from parallax.bitmamba import BitMambaConfig, BitMambaLM
from parallax.bitnet import quantize_weights_ternary
from parallax.gguf import load_gguf
from parallax.gguf_exporter import export_to_gguf_v3
from parallax.gguf_loader import load_gguf_pretrained, save_parallax_to_gguf
from parallax.gguf_tokenizer import GGUFTokenizer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_parallax_to_gguf_roundtrip_tied_weights():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        tie_weights=True,
    )
    model_orig = ParallaxLM(cfg)

    # Synthetic tokenizer
    tokens = [f"tok_{i}" for i in range(32)]
    tokens[0] = "<unk>"
    tokens[1] = "<s>"
    tokens[2] = "</s>"
    tokenizer = GGUFTokenizer(tokens=tokens, bos_token_id=1, eos_token_id=2)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        # Export ParallaxLM to standard GGUF v3
        save_parallax_to_gguf(model_orig, tmp_path, tokenizer=tokenizer)

        # Import back using load_gguf_pretrained
        model_loaded, tok_loaded = load_gguf_pretrained(tmp_path, return_tokenizer=True)

        assert model_loaded.config.vocab_size == cfg.vocab_size
        assert model_loaded.config.embed_dim == cfg.embed_dim
        assert model_loaded.config.num_layers == cfg.num_layers
        assert model_loaded.config.num_heads == cfg.num_heads
        assert model_loaded.config.tie_weights is True

        assert tok_loaded.vocab_size == 32
        assert tok_loaded.bos_token_id == 1
        assert tok_loaded.eos_token_id == 2

        # Verify exact weight match
        np.testing.assert_allclose(
            model_loaded.token_embeddings.weight.data,
            model_orig.token_embeddings.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model_loaded.blocks[0].attn.q_proj.weight.data,
            model_orig.blocks[0].attn.q_proj.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            model_loaded.final_norm.weight.data,
            model_orig.final_norm.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )

        # Verify identical inference output
        idx = np.array([[1, 5, 12, 24]], dtype=np.int64)
        logits_orig = model_orig(idx).data
        logits_loaded = model_loaded(idx).data

        np.testing.assert_allclose(logits_loaded, logits_orig, atol=1e-5, rtol=1e-5)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_parallax_to_gguf_roundtrip_untied_weights():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=20,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
        intermediate_size=16,
        tie_weights=False,
    )
    model_orig = ParallaxLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_parallax_to_gguf(model_orig, tmp_path)
        model_loaded = load_gguf_pretrained(tmp_path)

        assert model_loaded.config.tie_weights is False
        np.testing.assert_allclose(
            model_loaded.lm_head.weight.data,
            model_orig.lm_head.weight.data,
            atol=1e-6,
            rtol=1e-6,
        )

        idx = np.array([[1, 2, 3]], dtype=np.int64)
        np.testing.assert_allclose(
            model_loaded(idx).data,
            model_orig(idx).data,
            atol=1e-5,
            rtol=1e-5,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_parallax_to_gguf_roundtrip_swiglu():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        qkv_bias=True,
    )
    model_orig = ParallaxLM(cfg)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_parallax_to_gguf(model_orig, tmp_path)
        model_loaded = load_gguf_pretrained(tmp_path)

        assert model_loaded.config.use_swiglu is True
        assert model_loaded.config.num_kv_heads == 1
        assert model_loaded.config.rope_interleaved is False
        assert model_loaded.config.qkv_bias is True

        idx = np.array([[1, 5, 8]], dtype=np.int64)
        logits_orig = model_orig(idx).data
        logits_loaded = model_loaded(idx).data

        np.testing.assert_allclose(logits_loaded, logits_orig, atol=1e-5, rtol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
