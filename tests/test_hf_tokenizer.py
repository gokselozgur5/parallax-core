import json
import os
import tempfile
import numpy as np
import pytest

from parallax.generator import generate
from parallax.hf_loader import load_hf_pretrained
from parallax.hf_tokenizer import HFTokenizer, load_hf_tokenizer
from parallax.safetensors import save_safetensors


def test_hf_tokenizer_direct_bpe():
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "h": 4,
        "e": 5,
        "l": 6,
        "o": 7,
        "w": 8,
        "r": 9,
        "d": 10,
        "Ġ": 11,
        "he": 12,
        "ll": 13,
        "hello": 14,
        "Ġw": 15,
        "or": 16,
        "ld": 17,
        "world": 18,
        "Ġworld": 19,
    }
    merges = [
        "h e",
        "l l",
        "he ll",
        "hell o",
        "Ġ w",
        "o r",
        "l d",
        "w or",
        "wor ld",
        "Ġ world",
    ]
    added = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }

    tok = HFTokenizer(
        vocab=vocab,
        merges=merges,
        added_tokens=added,
        bos_token="<bos>",
        eos_token="<eos>",
    )

    assert tok.vocab_size >= 14
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2

    # Encode "hello world"
    ids = tok.encode("hello world")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded == "hello world"


def test_load_hf_tokenizer_from_json_and_pretrained_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Config.json
        cfg = {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "intermediate_size": 32,
            "tie_word_embeddings": True,
        }
        with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        # 2. Tokenizer.json
        tok_data = {
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": True},
                {"id": 1, "content": "<bos>", "special": True},
                {"id": 2, "content": "<eos>", "special": True},
            ],
            "model": {
                "type": "BPE",
                "vocab": {
                    "<pad>": 0,
                    "<bos>": 1,
                    "<eos>": 2,
                    "h": 3,
                    "i": 4,
                    "hi": 5,
                },
                "merges": ["h i"],
            },
        }
        with open(os.path.join(tmp_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tok_data, f)

        # 3. Model weights in safetensors
        tensors = {
            "model.embed_tokens.weight": np.random.randn(32, 16).astype(np.float32),
            "model.norm.weight": np.ones((16,), dtype=np.float32),
            "model.layers.0.input_layernorm.weight": np.ones((16,), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones((16,), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(16, 32).astype(np.float32),
        }
        save_safetensors(os.path.join(tmp_dir, "model.safetensors"), tensors)

        # Test standalone load_hf_tokenizer
        tokenizer = load_hf_tokenizer(tmp_dir)
        assert isinstance(tokenizer, HFTokenizer)
        assert tokenizer.vocab["hi"] == 5

        # Test load_hf_pretrained with return_tokenizer=True
        model, tok = load_hf_pretrained(tmp_dir, return_tokenizer=True)
        assert tok is not None

        # Run prompt generation pipeline
        prompt_ids = tok.encode("hi")
        out_ids = generate(model, prompt_ids, max_new_tokens=3, temperature=0.0)
        assert len(out_ids) == len(prompt_ids) + 3

        decoded = tok.decode(out_ids)
        assert isinstance(decoded, str)
