"""
Unit tests for Byte-Pair Encoding (BPE) Tokenizer.
"""

import os
import tempfile

from parallax.tokenizer import BPETokenizer, get_pair_stats, merge_tokens


def test_get_pair_stats_and_merge_tokens():
    tokens = [1, 2, 3, 1, 2, 4]
    stats = get_pair_stats(tokens)
    assert stats[(1, 2)] == 2
    assert stats[(2, 3)] == 1
    assert stats[(3, 1)] == 1

    merged = merge_tokens(tokens, (1, 2), 99)
    assert merged == [99, 3, 99, 4]


def test_bpe_training_and_reconstruction():
    text = "low lower newest lowest wide wider low lower lowest"
    tokenizer = BPETokenizer()
    initial_vocab_size = tokenizer.vocab_size

    # Train 10 merges
    target_vocab_size = initial_vocab_size + 10
    tokenizer.train(text, vocab_size=target_vocab_size)

    assert tokenizer.vocab_size == target_vocab_size
    assert len(tokenizer.merges) == 10

    # Encoding should be shorter than raw bytes length
    raw_bytes_len = len(text.encode("utf-8"))
    encoded = tokenizer.encode(text)
    assert len(encoded) < raw_bytes_len

    # Exact decoding reconstruction
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_bpe_handles_unicode_and_unseen_chars():
    text = "Paralaks derinlik kurar: 🌟 αβγ Türkçe karakterler: çğışöü"
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=275)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

    # Unseen text encoding & lossless decoding
    test_str = "Yeni bir test cümlesi! 🚀"
    enc_test = tokenizer.encode(test_str)
    dec_test = tokenizer.decode(enc_test)
    assert dec_test == test_str


def test_bpe_serialization_save_and_load():
    corpus = "the quick brown fox jumps over the lazy dog repeatedly"
    tok1 = BPETokenizer()
    tok1.train(corpus, vocab_size=270)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bpe_tokenizer.json")
        tok1.save(path)

        tok2 = BPETokenizer.load(path)
        assert tok2.vocab_size == tok1.vocab_size
        assert tok2.merges == tok1.merges

        sample = "the lazy dog jumps"
        assert tok1.encode(sample) == tok2.encode(sample)
        assert tok2.decode(tok2.encode(sample)) == sample


def test_bpe_special_tokens_preservation():
    tokenizer = BPETokenizer(special_tokens={"<bos>": 257, "<eos>": 258, "<pad>": 256})
    sample = "<bos>hello world<eos>"
    encoded = tokenizer.encode(sample)
    assert encoded[0] == 257
    assert encoded[-1] == 258
    decoded = tokenizer.decode(encoded)
    assert decoded == sample

