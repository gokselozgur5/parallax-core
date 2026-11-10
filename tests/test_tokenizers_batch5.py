import json
import os
import tempfile
import pytest

from parallax.gguf_tokenizer import GGUFTokenizer
from parallax.hf_tokenizer import HFTokenizer, load_hf_tokenizer


def test_gguf_tokenizer_byte_level_bpe_decoding_gpt2():
    tokens = [
        "<unk>",
        "<s>",
        "</s>",
        "\u0120world",
        "\u010A",
    ]
    tok = GGUFTokenizer(tokens=tokens, model_type="gpt2")
    assert tok.decode([3]) == " world"
    assert tok.decode([4]) == "\n"
    assert tok.decode([3, 4]) == " world\n"


def test_gguf_tokenizer_rank_ordered_bpe_and_spm_dummy_prefix():
    tokens = ["a", "b", "c", "ab", "bc"]
    merges = ["b c", "a b"]
    tok = GGUFTokenizer(tokens=tokens, merges=merges)
    encoded = tok.encode("abc")
    assert [tokens[i] for i in encoded] == ["a", "bc"]

    spm_tokens = [" ", "a", "b", " a", " b"]
    spm_merges = ["  a"]
    spm_tok = GGUFTokenizer(tokens=spm_tokens, merges=spm_merges, model_type="llama")
    encoded_spm = spm_tok.encode("a")
    assert spm_tok.decode(encoded_spm).startswith(" ")


def test_hf_tokenizer_merges_as_pairs():
    vocab = {"a": 0, "b": 1, "ab": 2}
    merges_str = ["a b"]
    merges_pair = [["a", "b"]]
    tok1 = HFTokenizer(vocab=vocab, merges=merges_str)
    tok2 = HFTokenizer(vocab=vocab, merges=merges_pair)
    assert tok1.merges == tok2.merges
    assert tok1.encode("ab") == tok2.encode("ab") == [2]


def test_hf_tokenizer_pretokenizer_split():
    vocab = {
        "I": 0,
        "'": 1,
        "m": 2,
        "'m": 3,
        "Ġ": 4,
        "1": 5,
        "0": 6,
        "10": 7,
        "100": 8,
        "Ġ100": 9,
        "%": 10,
        "s": 11,
        "u": 12,
        "r": 13,
        "e": 14,
        "su": 15,
        "sur": 16,
        "sure": 17,
        "Ġsure": 18,
        ",": 19,
        "a": 20,
        "n": 21,
        "ar": 22,
        "are": 23,
        "aren": 24,
        "Ġaren": 25,
        "t": 26,
        "'t": 27,
        "y": 28,
        "o": 29,
        "yo": 30,
        "you": 31,
        "Ġyou": 32,
        "?": 33,
    }
    merges = [
        "' m",
        "1 0",
        "10 0",
        "Ġ 100",
        "s u",
        "su r",
        "sur e",
        "Ġ sure",
        "a r",
        "ar e",
        "are n",
        "Ġ aren",
        "' t",
        "y o",
        "yo u",
        "Ġ you",
    ]
    tok = HFTokenizer(vocab=vocab, merges=merges)
    encoded = tok.encode("I'm 100% sure, aren't you?")
    assert encoded == [0, 3, 9, 10, 18, 19, 25, 27, 32, 33]
    decoded = tok.decode(encoded)
    assert decoded == "I'm 100% sure, aren't you?"


def test_hf_tokenizer_strict_unknown_and_config_error():
    vocab = {"a": 0, "b": 1}
    tok_no_unk = HFTokenizer(vocab=vocab, unk_token=None)
    with pytest.raises(KeyError, match="Token 'c' not found in vocabulary and no unk_token defined."):
        tok_no_unk.encode("abc")

    tok_with_unk = HFTokenizer(vocab=vocab, unk_token="<unk>")
    tok_with_unk.unk_token_id = 99
    assert tok_with_unk.encode("abc") == [0, 1, 99]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tok_file = os.path.join(tmp_dir, "tokenizer.json")
        cfg_file = os.path.join(tmp_dir, "tokenizer_config.json")
        with open(tok_file, "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": {"a": 0}}}, f)
        with open(cfg_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content")

        with pytest.raises(json.JSONDecodeError):
            load_hf_tokenizer(tmp_dir)
