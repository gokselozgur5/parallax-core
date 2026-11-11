import os
import tempfile
import numpy as np
import pytest

from parallax.generator import generate
from parallax.gguf import save_gguf
from parallax.gguf_loader import load_gguf_pretrained
from parallax.gguf_tokenizer import GGUFTokenizer, load_gguf_tokenizer


def test_gguf_tokenizer_encode_decode():
    tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "<0x20>",
        "h",
        "e",
        "l",
        "o",
        "w",
        "r",
        "d",
        "he",
        "ll",
        "lo",
        "hello",
        "world",
    ]
    merges = [
        "h e",
        "l l",
        "l o",
        "he lo",
    ]

    tok = GGUFTokenizer(
        tokens=tokens,
        merges=merges,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
    )

    assert tok.vocab_size == len(tokens)
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2

    # Test direct decode
    encoded = tok.encode("hello")
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == "hello"

    # Test byte fallback for space
    space_enc = tok.encode(" ")
    assert space_enc == [3]  # <0x20>
    assert tok.decode([3]) == " "


def test_gguf_tokenizer_from_gguf_file_and_model_inference():
    vocab_size = 64
    embed_dim = 16
    num_heads = 2
    num_kv_heads = 2
    num_layers = 1
    intermediate_size = 32

    tokens = [f"t_{i}" for i in range(vocab_size)]
    tokens[0] = "<unk>"
    tokens[1] = "<s>"
    tokens[2] = "</s>"
    tokens[3] = "hi"
    tokens[4] = "ai"

    metadata = {
        "general.architecture": "llama",
        "llama.embedding_length": embed_dim,
        "llama.block_count": num_layers,
        "llama.feed_forward_length": intermediate_size,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.freq_base": 10000.0,
        "llama.context_length": 128,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
    }

    tensors = {
        "token_embd.weight": np.random.randn(vocab_size, embed_dim).astype(np.float32),
        "output_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.attn_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.ffn_norm.weight": np.ones((embed_dim,), dtype=np.float32),
        "blk.0.attn_q.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_k.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_v.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.attn_output.weight": np.random.randn(embed_dim, embed_dim).astype(np.float32),
        "blk.0.ffn_gate.weight": np.random.randn(intermediate_size, embed_dim).astype(np.float32),
        "blk.0.ffn_up.weight": np.random.randn(intermediate_size, embed_dim).astype(np.float32),
        "blk.0.ffn_down.weight": np.random.randn(embed_dim, intermediate_size).astype(np.float32),
    }

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_gguf(tmp_path, metadata=metadata, tensors=tensors, version=3)

        # Test standalone load_gguf_tokenizer
        tokenizer = load_gguf_tokenizer(tmp_path)
        assert tokenizer.vocab_size == vocab_size
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2

        # Test paired load_gguf_pretrained
        model, tok = load_gguf_pretrained(tmp_path, return_tokenizer=True)
        assert tok.vocab_size == vocab_size

        # End-to-end prompt encoding, generation, and decoding
        prompt_ids = tok.encode("hi")
        generated_ids = generate(model, prompt_ids, max_new_tokens=3, temperature=0.0, use_cache=True)
        assert len(generated_ids) == len(prompt_ids) + 3

        decoded_text = tok.decode(generated_ids)
        assert isinstance(decoded_text, str)
        assert len(decoded_text) > 0

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_tokenizer_out_of_bounds_special_token_ids():
    """Verify special token IDs >= vocab_size or < 0 are set to None."""
    tokens = ["<unk>", "<s>", "</s>", "a", "b"]
    tok = GGUFTokenizer(
        tokens=tokens,
        bos_token_id=999,
        eos_token_id=-1,
        pad_token_id=10,
        unk_token_id=100,
    )
    assert tok.bos_token_id is None
    assert tok.eos_token_id is None
    assert tok.pad_token_id is None
    assert tok.unk_token_id is None
    # Encoding with unk=None does not raise IndexError
    encoded = tok.encode("z")
    assert isinstance(encoded, list)
