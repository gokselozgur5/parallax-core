import os
import struct
import tempfile
import numpy as np
import pytest

from parallax.eval import compute_perplexity
from parallax.gguf import (
    GGML_TYPE_F32,
    GGUF_MAGIC,
    load_gguf,
)
from parallax.safetensors import load_safetensors


class BigramDummyModel:
    """Deterministic bigram LM where next-token logits depend only on previous token.

    This ensures full-sequence perplexity and sliding-window perplexity match exactly
    regardless of window size or stride.
    """

    def __init__(self, vocab_size: int = 16, max_seq_len: int = 16):
        self.vocab_size = vocab_size
        np.random.seed(42)
        # Random transition matrix [vocab_size, vocab_size]
        self.table = np.random.randn(vocab_size, vocab_size).astype(np.float64)

        class Config:
            pass

        self.config = Config()
        self.config.max_seq_len = max_seq_len

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        # inputs: [1, seq_len]
        seq = inputs[0]
        # Return logits for each position based only on current token
        logits = self.table[seq]  # [seq_len, vocab_size]
        return logits[None, :, :]  # [1, seq_len, vocab_size]


def test_sliding_window_perplexity_no_skipped_targets():
    """Issue #277: Sliding window perplexity must not skip targets across window boundaries."""
    vocab_size = 16
    model = BigramDummyModel(vocab_size=vocab_size, max_seq_len=8)

    # Deterministic test token sequence
    np.random.seed(123)
    tokens = np.random.randint(0, vocab_size, size=24).tolist()

    # Full sequence perplexity (window_size >= seq_len)
    ppl_full = compute_perplexity(model, tokens, max_context_len=24, stride=24)

    # Various sliding window configurations
    ppl_w8_s4 = compute_perplexity(model, tokens, max_context_len=8, stride=4)
    ppl_w8_s2 = compute_perplexity(model, tokens, max_context_len=8, stride=2)
    ppl_w8_s1 = compute_perplexity(model, tokens, max_context_len=8, stride=1)
    ppl_w6_s3 = compute_perplexity(model, tokens, max_context_len=6, stride=3)

    # For bigram model, sliding PPL must match full PPL exactly across all strides
    np.testing.assert_allclose(ppl_w8_s4, ppl_full, rtol=1e-6)
    np.testing.assert_allclose(ppl_w8_s2, ppl_full, rtol=1e-6)
    np.testing.assert_allclose(ppl_w8_s1, ppl_full, rtol=1e-6)
    np.testing.assert_allclose(ppl_w6_s3, ppl_full, rtol=1e-6)


def test_gguf_tensor_bounds_validation():
    """Issue #301: load_gguf must raise ValueError on out-of-bounds offsets or lengths."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        alignment = 32
        with open(tmp_path, "wb") as f:
            # Header: magic (4), version (4), tensor_count (8), metadata_count (8)
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<IQQ", 3, 1, 0))

            # Tensor descriptor with excessive offset
            name_bytes = "oob_tensor".encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", 1))  # 1 dim
            f.write(struct.pack("<Q", 4))  # len 4 float32 = 16 bytes
            f.write(struct.pack("<IQ", GGML_TYPE_F32, 999999))  # large offset

            header_len = f.tell()
            pad = (alignment - (header_len % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)
            f.write(b"\x00" * 16)  # only 16 bytes payload

        with pytest.raises(ValueError, match="Invalid tensor offset/length: exceeds file bounds"):
            load_gguf(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_unsupported_tensor_type_raises():
    """Issue #302: load_gguf must raise NotImplementedError on unsupported GGML tensor types."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        alignment = 32
        unsupported_type = 99  # arbitrary unsupported GGML type
        with open(tmp_path, "wb") as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<IQQ", 3, 1, 0))

            name_bytes = "unsupported_tensor".encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<Q", 32))
            f.write(struct.pack("<IQ", unsupported_type, 0))

            header_len = f.tell()
            pad = (alignment - (header_len % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)
            f.write(b"\x00" * 64)

        with pytest.raises(NotImplementedError, match=f"Unsupported GGML tensor type: {unsupported_type}"):
            load_gguf(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_safetensors_bounds_validation():
    """Issue #301: load_safetensors must validate header and tensor bounds against file size."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_path = f.name

    try:
        # 1. Truncated header length
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", 1000000))  # Claim 1MB header
            f.write(b'{"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}')

        with pytest.raises(ValueError, match="Truncated header in safetensors file"):
            load_safetensors(tmp_path)

        # 2. Out of bounds data offsets
        import json
        header_dict = {
            "t": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 1000000],  # Out of bounds payload
            }
        }
        header_bytes = json.dumps(header_dict).encode("utf-8")
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b"\x00" * 4)

        with pytest.raises(ValueError, match="Invalid tensor offset/length: exceeds file bounds"):
            load_safetensors(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
