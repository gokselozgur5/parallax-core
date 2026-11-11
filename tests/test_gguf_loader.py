import os
import tempfile
import numpy as np
import pytest

from parallax.generator import generate
from parallax.gguf import save_gguf
from parallax.gguf_loader import (
    gguf_metadata_to_parallax,
    load_gguf_pretrained,
    load_gguf_weights_into_model,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.tensor import Tensor


def test_gguf_metadata_to_parallax_config():
    metadata = {
        "general.architecture": "llama",
        "llama.embedding_length": 64,
        "llama.block_count": 2,
        "llama.feed_forward_length": 128,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "llama.attention.layer_norm_rms_epsilon": 1e-6,
        "llama.rope.freq_base": 500000.0,
        "llama.context_length": 1024,
        "tokenizer.ggml.tokens": [f"tok_{i}" for i in range(256)],
    }

    tensors = {"output.weight": np.zeros((256, 64), dtype=np.float32)}

    cfg = gguf_metadata_to_parallax(metadata, tensors)
    assert cfg.vocab_size == 256
    assert cfg.embed_dim == 64
    assert cfg.num_heads == 4
    assert cfg.num_kv_heads == 2
    assert cfg.num_layers == 2
    assert cfg.intermediate_size == 128
    assert cfg.rms_norm_eps == 1e-6
    assert cfg.rope_theta == 500000.0
    assert cfg.max_seq_len == 1024
    assert cfg.tie_weights is False
    assert cfg.qkv_bias is False


def test_load_gguf_pretrained_end_to_end():
    vocab_size = 128
    embed_dim = 32
    num_heads = 4
    num_kv_heads = 2
    num_layers = 2
    intermediate_size = 64
    head_dim = embed_dim // num_heads

    metadata = {
        "general.architecture": "llama",
        "llama.embedding_length": embed_dim,
        "llama.block_count": num_layers,
        "llama.feed_forward_length": intermediate_size,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.freq_base": 10000.0,
        "llama.context_length": 512,
        "tokenizer.ggml.tokens": [f"token_{i}" for i in range(vocab_size)],
    }

    tensors = {
        "token_embd.weight": np.random.randn(vocab_size, embed_dim).astype(np.float32),
        "output_norm.weight": np.ones((embed_dim,), dtype=np.float32),
    }

    for i in range(num_layers):
        tensors[f"blk.{i}.attn_norm.weight"] = np.ones((embed_dim,), dtype=np.float32)
        tensors[f"blk.{i}.ffn_norm.weight"] = np.ones((embed_dim,), dtype=np.float32)
        tensors[f"blk.{i}.attn_q.weight"] = np.random.randn(num_heads * head_dim, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.attn_k.weight"] = np.random.randn(num_kv_heads * head_dim, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.attn_v.weight"] = np.random.randn(num_kv_heads * head_dim, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.attn_output.weight"] = np.random.randn(embed_dim, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.ffn_gate.weight"] = np.random.randn(intermediate_size, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.ffn_up.weight"] = np.random.randn(intermediate_size, embed_dim).astype(np.float32)
        tensors[f"blk.{i}.ffn_down.weight"] = np.random.randn(embed_dim, intermediate_size).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_gguf(tmp_path, metadata=metadata, tensors=tensors, version=3)

        model = load_gguf_pretrained(tmp_path)
        assert isinstance(model, ParallaxLM)
        assert model.config.vocab_size == vocab_size
        assert model.config.embed_dim == embed_dim
        assert model.config.num_layers == num_layers
        assert model.config.tie_weights is True

        # Verify weights were correctly loaded
        np.testing.assert_allclose(
            model.token_embeddings.weight.data,
            tensors["token_embd.weight"],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            model.blocks[0].attn.q_proj.weight.data,
            tensors["blk.0.attn_q.weight"],
            rtol=1e-5,
            atol=1e-5,
        )

        # Run forward pass inference
        idx = Tensor(np.array([[1, 5, 10, 42]], dtype=np.int64))
        logits = model(idx)
        assert logits.shape == (1, 4, vocab_size)
        assert not np.isnan(logits.data).any()

        # Autoregressive generation with KV cache
        prompt = [1, 5]
        out = generate(model, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
        assert len(out) == len(prompt) + 4
        assert all(0 <= tok < vocab_size for tok in out)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_gguf_qwen2_qkv_bias_strict_no_attn_output_bias():
    vocab_size = 64
    embed_dim = 16
    num_heads = 2
    num_layers = 1
    intermediate_size = 32

    metadata = {
        "general.architecture": "qwen2",
        "qwen2.embedding_length": embed_dim,
        "qwen2.block_count": num_layers,
        "qwen2.feed_forward_length": intermediate_size,
        "qwen2.attention.head_count": num_heads,
        "qwen2.attention.head_count_kv": num_heads,
        "qwen2.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen2.rope.freq_base": 10000.0,
        "qwen2.context_length": 64,
        "tokenizer.ggml.tokens": [f"tok_{i}" for i in range(vocab_size)],
    }

    tensors = {
        "token_embd.weight": np.random.randn(vocab_size, embed_dim).astype(np.float32),
        "output_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.attn_q.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_q.bias": np.zeros((embed_dim,), dtype=np.float32),
        "blk.0.attn_k.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_k.bias": np.zeros((embed_dim,), dtype=np.float32),
        "blk.0.attn_v.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_v.bias": np.zeros((embed_dim,), dtype=np.float32),
        "blk.0.attn_output.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        # Note: blk.0.attn_output.bias is absent, exactly like real Qwen2 GGUF checkpoints!
        "blk.0.ffn_gate.weight": np.random.randn(intermediate_size, embed_dim).astype(np.float32),
        "blk.0.ffn_up.weight": np.random.randn(intermediate_size, embed_dim).astype(np.float32),
        "blk.0.ffn_down.weight": np.random.randn(embed_dim, intermediate_size).astype(np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name
    try:
        save_gguf(tmp_path, metadata=metadata, tensors=tensors, version=3)
        model = load_gguf_pretrained(tmp_path, strict=True)
        assert model.config.qkv_bias is True
        assert model.blocks[0].attn.out_proj.bias is None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

