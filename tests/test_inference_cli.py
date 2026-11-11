"""
Regression tests for CLI inference and incremental UTF-8 streaming decoding.
"""


from parallax.generator import CharTokenizer
from parallax.tokenizer import BPETokenizer, IncrementalTokenDecoder


def test_incremental_token_decoder_splits_multibyte_utf8():
    tokenizer = BPETokenizer()
    text = "çğıöşü İstanbul 🚀"
    token_ids = tokenizer.encode(text)

    # Per-token decode without incremental decoder produces replacement characters (U+FFFD)
    naive_decoded = "".join([tokenizer.decode([t]) for t in token_ids])
    assert "\ufffd" in naive_decoded

    # IncrementalTokenDecoder reconstructs exact Unicode string without U+FFFD
    stream_decoder = IncrementalTokenDecoder(tokenizer)
    streamed_pieces = [stream_decoder.decode_token(t) for t in token_ids]
    streamed_pieces.append(stream_decoder.finish())
    streamed_decoded = "".join(streamed_pieces)

    assert "\ufffd" not in streamed_decoded
    assert streamed_decoded == text
    assert streamed_decoded == tokenizer.decode(token_ids)


def test_incremental_token_decoder_fallback_and_none_tokenizer():
    # Without tokenizer, formats [token_id]
    no_tok_decoder = IncrementalTokenDecoder(None)
    assert no_tok_decoder.decode_token(42) == "[42] "
    assert no_tok_decoder.finish() == ""

    # With CharTokenizer (decode returns str)
    char_tok = CharTokenizer("Hello 世界")
    char_decoder = IncrementalTokenDecoder(char_tok)
    encoded = char_tok.encode("Hello 世界")
    result = "".join([char_decoder.decode_token(t) for t in encoded]) + char_decoder.finish()
    assert result == "Hello 世界"


def test_incremental_token_decoder_incomplete_trailing_bytes():
    class DummyByteTokenizer:
        def __init__(self):
            self.id_to_bytes = {
                1: b"\xc3",  # Half of 2-byte UTF-8 sequence
            }

    tok = DummyByteTokenizer()
    decoder = IncrementalTokenDecoder(tok)
    part1 = decoder.decode_token(1)
    assert part1 == ""  # Incomplete byte buffered
    finished = decoder.finish()
    assert finished == "\ufffd"  # Flushed incomplete bytes become replacement char
