import os
import tempfile
import numpy as np
import pytest

from parallax.gguf_loader import (
    gguf_metadata_to_parallax,
    load_gguf_pretrained,
    save_parallax_to_gguf,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.tensor import Tensor


def test_gguf_metadata_missing_required_keys():
    # Issue #226: missing embedding_length, block_count, or head_count must raise ValueError
    base_meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 64,
        "llama.block_count": 2,
        "llama.attention.head_count": 4,
    }

    # Verify base succeeds
    cfg = gguf_metadata_to_parallax(base_meta)
    assert cfg.embed_dim == 64
    assert cfg.num_layers == 2
    assert cfg.num_heads == 4

    # Test missing embedding_length
    meta_no_embed = dict(base_meta)
    del meta_no_embed["llama.embedding_length"]
    with pytest.raises(ValueError, match="Missing required GGUF metadata key: 'llama.embedding_length'"):
        gguf_metadata_to_parallax(meta_no_embed)

    # Test missing block_count
    meta_no_blocks = dict(base_meta)
    del meta_no_blocks["llama.block_count"]
    with pytest.raises(ValueError, match="Missing required GGUF metadata key: 'llama.block_count'"):
        gguf_metadata_to_parallax(meta_no_blocks)

    # Test missing head_count
    meta_no_heads = dict(base_meta)
    del meta_no_heads["llama.attention.head_count"]
    with pytest.raises(ValueError, match="Missing required GGUF metadata key: 'llama.attention.head_count'"):
        gguf_metadata_to_parallax(meta_no_heads)


def test_gguf_gelu_mlp_bias_roundtrip():
    # Issue #231: non-SwiGLU GELU MLP with biases round-trips through save & load
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=128,
        use_swiglu=False,
        intermediate_size=64,
        tie_weights=False,
    )
    model = ParallaxLM(cfg)

    # Ensure fc1 and fc2 have biases initialized
    for block in model.blocks:
        assert hasattr(block.mlp, "fc1")
        assert hasattr(block.mlp, "fc2")
        block.mlp.fc1.bias.data = np.random.randn(*block.mlp.fc1.bias.shape).astype(np.float64)
        block.mlp.fc2.bias.data = np.random.randn(*block.mlp.fc2.bias.shape).astype(np.float64)

    idx = Tensor(np.array([[1, 4, 15, 23]], dtype=np.int64))
    orig_logits = model(idx).data

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_parallax_to_gguf(model, tmp_path, architecture="llama")
        loaded_model = load_gguf_pretrained(tmp_path)

        for i, (orig_blk, load_blk) in enumerate(zip(model.blocks, loaded_model.blocks)):
            assert load_blk.mlp.fc1.bias is not None, f"Block {i} fc1 bias missing"
            assert load_blk.mlp.fc2.bias is not None, f"Block {i} fc2 bias missing"
            np.testing.assert_allclose(
                load_blk.mlp.fc1.bias.data,
                orig_blk.mlp.fc1.bias.data,
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                load_blk.mlp.fc2.bias.data,
                orig_blk.mlp.fc2.bias.data,
                rtol=1e-5,
                atol=1e-5,
            )

        loaded_logits = loaded_model(idx).data
        np.testing.assert_allclose(orig_logits, loaded_logits, rtol=1e-5, atol=1e-5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_rope_interleaved_default_and_override():
    # Issue #233: default to interleaved for llama/mistral, contiguous for qwen2
    llama_meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 64,
        "llama.block_count": 2,
        "llama.attention.head_count": 4,
    }
    cfg_llama = gguf_metadata_to_parallax(llama_meta)
    assert cfg_llama.rope_interleaved is True

    mistral_meta = {
        "general.architecture": "mistral",
        "mistral.embedding_length": 64,
        "mistral.block_count": 2,
        "mistral.attention.head_count": 4,
    }
    cfg_mistral = gguf_metadata_to_parallax(mistral_meta)
    assert cfg_mistral.rope_interleaved is True

    qwen2_meta = {
        "general.architecture": "qwen2",
        "qwen2.embedding_length": 64,
        "qwen2.block_count": 2,
        "qwen2.attention.head_count": 4,
    }
    cfg_qwen2 = gguf_metadata_to_parallax(qwen2_meta)
    assert cfg_qwen2.rope_interleaved is False

    # Explicit override with key
    llama_explicit_false = dict(llama_meta)
    llama_explicit_false["llama.rope.interleaved"] = False
    assert gguf_metadata_to_parallax(llama_explicit_false).rope_interleaved is False

    qwen2_explicit_true = dict(qwen2_meta)
    qwen2_explicit_true["qwen2.rope.interleaved"] = True
    assert gguf_metadata_to_parallax(qwen2_explicit_true).rope_interleaved is True
