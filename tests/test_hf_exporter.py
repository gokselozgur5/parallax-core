import os
import tempfile
import numpy as np

from parallax.gguf_loader import load_gguf_pretrained, save_parallax_to_gguf
from parallax.hf_loader import load_hf_pretrained, save_hf_pretrained
from parallax.hf_tokenizer import HFTokenizer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_hf_save_and_load_roundtrip():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
        use_swiglu=True,
        rope_interleaved=False,
        tie_weights=True,
        qkv_bias=True,
    )
    model_orig = ParallaxLM(cfg)

    vocab = {f"tok_{i}": i for i in range(3, 32)}
    vocab["<pad>"] = 0
    vocab["<bos>"] = 1
    vocab["<eos>"] = 2
    tok_orig = HFTokenizer(vocab=vocab, bos_token="<bos>", eos_token="<eos>")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save ParallaxLM directly into HuggingFace format
        save_hf_pretrained(model_orig, tmp_dir, tokenizer=tok_orig, model_type="llama")

        assert os.path.exists(os.path.join(tmp_dir, "config.json"))
        assert os.path.exists(os.path.join(tmp_dir, "model.safetensors"))
        assert os.path.exists(os.path.join(tmp_dir, "tokenizer.json"))

        # Load back with zero dependencies
        model_loaded, tok_loaded = load_hf_pretrained(tmp_dir, return_tokenizer=True)

        assert model_loaded.config.embed_dim == cfg.embed_dim
        assert model_loaded.config.num_layers == cfg.num_layers
        assert model_loaded.config.num_heads == cfg.num_heads
        assert model_loaded.config.tie_weights is True
        assert model_loaded.config.qkv_bias is True

        assert tok_loaded is not None
        assert tok_loaded.vocab_size == 32

        # Verify weights match
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

        # Verify identical inference forward pass
        idx = np.array([[1, 4, 15, 29]], dtype=np.int64)
        logits_orig = model_orig(idx).data
        logits_loaded = model_loaded(idx).data

        np.testing.assert_allclose(logits_loaded, logits_orig, atol=1e-5, rtol=1e-5)


def test_bidirectional_gguf_to_hf_bridge():
    """Validates cross-format conversion: ParallaxLM -> GGUF -> ParallaxLM -> SafeTensors -> ParallaxLM."""
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=20,
        embed_dim=16,
        num_heads=2,
        num_kv_heads=2,
        num_layers=1,
        max_seq_len=32,
        intermediate_size=16,
        use_swiglu=True,
        rope_interleaved=False,
        tie_weights=False,
    )
    model_orig = ParallaxLM(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        gguf_path = os.path.join(tmp_dir, "model.gguf")
        hf_dir = os.path.join(tmp_dir, "hf_export")

        # 1. ParallaxLM -> GGUF
        save_parallax_to_gguf(model_orig, gguf_path)

        # 2. GGUF -> ParallaxLM
        model_from_gguf = load_gguf_pretrained(gguf_path)

        # 3. ParallaxLM -> HuggingFace SafeTensors
        save_hf_pretrained(model_from_gguf, hf_dir, model_type="llama")

        # 4. SafeTensors -> ParallaxLM
        model_from_hf = load_hf_pretrained(hf_dir)

        # 5. Verify identical inference across the entire cross-conversion bridge
        idx = np.array([[1, 2, 3]], dtype=np.int64)
        logits_orig = model_orig(idx).data
        logits_final = model_from_hf(idx).data

        np.testing.assert_allclose(logits_final, logits_orig, atol=1e-5, rtol=1e-5)


def test_hf_save_and_load_tokenizer_split_pattern_roundtrip():
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE

    qwen_pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    vocab = {chr(i): i for i in range(256)}
    for b, u in _BYTE_TO_UNICODE.items():
        vocab[u] = len(vocab)

    tok_orig = HFTokenizer(vocab=vocab, split_pattern=qwen_pat)
    cfg = ParallaxLMConfig(
        vocab_size=len(vocab), embed_dim=16, num_heads=2, num_kv_heads=2,
        num_layers=1, max_seq_len=32, intermediate_size=16, use_swiglu=True,
    )
    model = ParallaxLM(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_hf_pretrained(model, tmp_dir, tokenizer=tok_orig)
        _, tok_loaded = load_hf_pretrained(tmp_dir, return_tokenizer=True)

        assert tok_loaded is not None
        assert tok_loaded.split_pattern == qwen_pat

        test_sentences = ["12345", "çok güzel", "naïve café"]
        for s in test_sentences:
            assert tok_loaded.encode(s) == tok_orig.encode(s)


def test_hf_save_sliding_window_and_normalizer():
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE

    vocab = {chr(i): i for i in range(256)}
    for b, u in _BYTE_TO_UNICODE.items():
        vocab[u] = len(vocab)

    tok_orig = HFTokenizer(vocab=vocab, normalizer="NFC")
    cfg = ParallaxLMConfig(
        vocab_size=len(vocab), embed_dim=16, num_heads=2, num_kv_heads=2,
        num_layers=1, max_seq_len=32, intermediate_size=16, use_swiglu=True,
        sliding_window=2048,
    )
    model = ParallaxLM(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_hf_pretrained(model, tmp_dir, tokenizer=tok_orig)
        model_loaded, tok_loaded = load_hf_pretrained(tmp_dir, return_tokenizer=True)

        assert model_loaded.config.sliding_window == 2048
        assert tok_loaded is not None
        assert tok_loaded.normalizer == "NFC"


