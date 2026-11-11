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
            "model_type": "llama",
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


def test_hf_tokenizer_pretokenizer_qwen_and_llama3_patterns():
    qwen_pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    llama3_pat = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    # Simple vocab containing byte tokens
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE
    vocab = {chr(i): i for i in range(256)}
    for b, u in _BYTE_TO_UNICODE.items():
        vocab[u] = len(vocab)

    tok_qwen = HFTokenizer(vocab=vocab, split_pattern=qwen_pat)
    tok_llama3 = HFTokenizer(vocab=vocab, split_pattern=llama3_pat)

    # Test digits chunking differences:
    # Qwen splits single digits
    q_matches = [m.group(0) for m in tok_qwen.pretokenizer_regex.finditer("12345")]
    assert q_matches == ["1", "2", "3", "4", "5"]

    # Llama 3 splits up to 3 digits
    l_matches = [m.group(0) for m in tok_llama3.pretokenizer_regex.finditer("12345")]
    assert l_matches == ["123", "45"]

    # Test Turkish and non-ASCII Unicode letter preservation
    tr_matches = [m.group(0) for m in tok_qwen.pretokenizer_regex.finditer("çok güzel")]
    assert tr_matches == ["çok", " güzel"]

    na_matches = [m.group(0) for m in tok_qwen.pretokenizer_regex.finditer("naïve café")]
    assert na_matches == ["naïve", " café"]


def test_hf_tokenizer_single_non_ascii_and_normalizer():
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE
    vocab = {u: b for b, u in _BYTE_TO_UNICODE.items()}
    c_bytes = "ç".encode("utf-8")
    c_bpe_str = "".join(_BYTE_TO_UNICODE[b] for b in c_bytes)
    vocab[c_bpe_str] = len(vocab)
    merges = [f"{_BYTE_TO_UNICODE[c_bytes[0]]} {_BYTE_TO_UNICODE[c_bytes[1]]}"]

    tok = HFTokenizer(vocab=vocab, merges=merges, normalizer="NFC")
    encoded = tok.encode("ç")
    assert tok.decode(encoded) == "ç"

    import unicodedata
    decomposed = unicodedata.normalize("NFD", "ç")
    assert tok.encode(decomposed) == tok.encode("ç")


def test_hf_tokenizer_unicode_categories_pattern():
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE

    pat = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}{1,3}|\p{P}+|\P{L}+"
    vocab = {chr(i): i for i in range(256)}
    for b, u in _BYTE_TO_UNICODE.items():
        vocab[u] = len(vocab)

    tok = HFTokenizer(vocab=vocab, split_pattern=pat)
    matches = [m.group(0) for m in tok.pretokenizer_regex.finditer("Hello world! 123")]
    assert "Hello" in matches


def test_hf_tokenizer_unsupported_pattern_fallback(monkeypatch):
    import warnings, builtins, pytest
    from parallax.hf_tokenizer import _PRETOKENIZER_REGEX, _translate_hf_regex

    with pytest.raises(ValueError, match="Unsupported regex construct"):
        _translate_hf_regex(r"[[:alpha:]]+")
    with pytest.raises(ValueError, match="Unsupported regex construct"):
        _translate_hf_regex(r"[a&&b]")
    with pytest.raises(ValueError, match="Unsupported Unicode property"):
        _translate_hf_regex(r"\p{NoSuchCategory}")

    orig_import = builtins.__import__
    def mock_import(name, *a, **kw):
        if name == "regex": raise ImportError
        return orig_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", mock_import)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok = HFTokenizer(vocab={"a": 0}, split_pattern=r"\p{NoSuchCategory}")
        assert len(w) == 1
        assert "Unsupported pre-tokenizer pattern" in str(w[0].message)
    assert tok.pretokenizer_regex == _PRETOKENIZER_REGEX


def test_hf_tokenizer_unsupported_sequence_fallback(tmp_path):
    import json, warnings
    from parallax.hf_tokenizer import _PRETOKENIZER_REGEX

    tok_file = tmp_path / "tokenizer.json"
    tok_data = {
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"\d+"}},
                {"type": "Split", "pattern": {"Regex": r"\s+"}},
            ],
        },
    }
    tok_file.write_text(json.dumps(tok_data))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tok = load_hf_tokenizer(str(tok_file))
        assert len(w) == 1
        assert "Unsupported Split configuration" in str(w[0].message)
    assert tok.pretokenizer_regex == _PRETOKENIZER_REGEX




def test_hf_tokenizer_load_split_pattern_encode_reference_ids():
    """Verify load_hf_tokenizer extracts Split pattern from tokenizer.json and encodes reference IDs (closes #437)."""
    qwen_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE
    vocab = {u: b for b, u in _BYTE_TO_UNICODE.items()}
    vocab["12"] = 1000
    vocab["123"] = 1001
    merges = ["1 2", "12 3"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tok_data = {
            "version": "1.0",
            "model": {"type": "BPE", "vocab": vocab, "merges": merges},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": qwen_pattern}, "behavior": "Isolated", "invert": False},
                    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
                ],
            },
        }
        with open(os.path.join(tmp_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tok_data, f)

        tok = load_hf_tokenizer(tmp_dir)
        # Because Qwen pattern splits each digit individually, '12' and '123' merges never form across chunks
        assert tok.encode("123") == [ord("1"), ord("2"), ord("3")]


def test_hf_tokenizer_vocab_size_with_added_tokens():
    """Verify vocab_size reflects added_tokens."""
    vocab = {"a": 0, "b": 1}
    added = {"<special>": 2}
    tok = HFTokenizer(vocab=vocab, added_tokens=added)
    assert tok.vocab_size == 3
    assert tok.vocab["<special>"] == 2


def test_hf_tokenizer_normalizer_preserves_added_tokens():
    """Verify normalizer only applies to non-special parts."""
    from parallax.hf_tokenizer import _BYTE_TO_UNICODE
    vocab = {u: b for b, u in _BYTE_TO_UNICODE.items()}
    vocab["Ã©"] = 300
    added = {"<reserved_special>": 301}
    # NFC normalizer turns e + \u0301 into é (which is bytes C3 A9 -> Ã© in byte-to-unicode)
    tok = HFTokenizer(vocab=vocab, added_tokens=added, normalizer="NFC", merges=["Ã ©"])
    encoded = tok.encode("<reserved_special>e\u0301")
    assert encoded[0] == 301
    assert encoded[1] == 300


def test_hf_tokenizer_merges_rstrip():
    """Verify rstrip on merge lines avoids breaking space tokens."""
    merges = ["Ġ a\n", "Ġ a\r\n"]
    tok = HFTokenizer(vocab={"Ġ": 0, "a": 1, "Ġa": 2}, merges=merges)
    assert ("Ġ", "a") in tok.merges
