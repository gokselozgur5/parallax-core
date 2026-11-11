import os
import struct
import tempfile
import numpy as np
import pytest

from parallax.gguf import GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGUF_MAGIC, load_gguf
from parallax.gguf_loader import load_gguf_pretrained
from parallax.gguf_quant import (
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from parallax.models.gpt import ParallaxLM


def test_q8_0_quantize_dequantize_roundtrip():
    np.random.seed(42)
    original = np.random.randn(2, 64).astype(np.float32)
    q_bytes = quantize_q8_0(original)
    dequant = dequantize_q8_0(q_bytes, original.shape)

    assert dequant.shape == original.shape
    # Q8_0 has 8-bit precision (~1/127 resolution), relative error within 3%
    np.testing.assert_allclose(dequant, original, atol=0.05, rtol=0.05)


def test_q4_0_quantize_dequantize_roundtrip():
    np.random.seed(42)
    original = np.random.randn(2, 64).astype(np.float32)
    q_bytes = quantize_q4_0(original)
    dequant = dequantize_q4_0(q_bytes, original.shape)

    assert dequant.shape == original.shape
    # Q4_0 has 4-bit precision (~1/8 resolution), test reasonable approximation
    corr = np.corrcoef(original.flatten(), dequant.flatten())[0, 1]
    assert corr > 0.95


def test_load_gguf_with_q8_0_and_q4_0_tensors():
    # Construct a synthetic GGUF v3 file containing Q8_0 and Q4_0 tensors
    shape_q8 = (32, 32)
    shape_q4 = (16, 32)

    raw_q8_orig = np.random.randn(*shape_q8).astype(np.float32)
    raw_q4_orig = np.random.randn(*shape_q4).astype(np.float32)

    bytes_q8 = quantize_q8_0(raw_q8_orig)
    bytes_q4 = quantize_q4_0(raw_q4_orig)

    alignment = 32

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        with open(tmp_path, "wb") as f:
            # Header: magic (4), version (4), tensor_count (8), metadata_count (8)
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<IQQ", 3, 2, 0))

            # Tensor descriptors
            # Tensor 1: q8_weight
            n1 = "q8_weight".encode("utf-8")
            f.write(struct.pack("<Q", len(n1)))
            f.write(n1)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<QQ", shape_q8[1], shape_q8[0]))  # inner-to-outer
            f.write(struct.pack("<IQ", GGML_TYPE_Q8_0, 0))

            # Offset 2 aligned
            pad1 = (alignment - (len(bytes_q8) % alignment)) % alignment
            offset2 = len(bytes_q8) + pad1

            # Tensor 2: q4_weight
            n2 = "q4_weight".encode("utf-8")
            f.write(struct.pack("<Q", len(n2)))
            f.write(n2)
            f.write(struct.pack("<I", 2))
            f.write(struct.pack("<QQ", shape_q4[1], shape_q4[0]))  # inner-to-outer
            f.write(struct.pack("<IQ", GGML_TYPE_Q4_0, offset2))

            # Align header
            header_len = f.tell()
            pad_h = (alignment - (header_len % alignment)) % alignment
            if pad_h > 0:
                f.write(b"\x00" * pad_h)

            # Write tensor payload
            f.write(bytes_q8)
            if pad1 > 0:
                f.write(b"\x00" * pad1)
            f.write(bytes_q4)

        # Load with automatic dequantization
        gguf_file = load_gguf(tmp_path, cast_float64=False)

        assert "q8_weight" in gguf_file.tensors
        assert "q4_weight" in gguf_file.tensors

        t_q8 = gguf_file.tensors["q8_weight"]
        t_q4 = gguf_file.tensors["q4_weight"]

        assert t_q8.shape == shape_q8
        assert t_q4.shape == shape_q4
        assert not np.isnan(t_q8).any()
        assert not np.isnan(t_q4).any()

        # Check values
        np.testing.assert_allclose(t_q8, raw_q8_orig, atol=0.05, rtol=0.05)
        corr4 = np.corrcoef(raw_q4_orig.flatten(), t_q4.flatten())[0, 1]
        assert corr4 > 0.95

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_load_gguf_pretrained_with_q8_weights():
    vocab_size = 64
    embed_dim = 32
    num_heads = 2
    num_kv_heads = 2
    num_layers = 1
    intermediate_size = 32

    tokens = [f"t_{i}" for i in range(vocab_size)]

    metadata = {
        "general.architecture": "llama",
        "llama.embedding_length": embed_dim,
        "llama.block_count": num_layers,
        "llama.feed_forward_length": intermediate_size,
        "llama.attention.head_count": num_heads,
        "llama.attention.head_count_kv": num_kv_heads,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.freq_base": 10000.0,
        "llama.context_length": 64,
        "tokenizer.ggml.tokens": tokens,
    }

    raw_tensors = {
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

    alignment = 32
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        with open(tmp_path, "wb") as f:
            # Magic & Header
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<IQQ", 3, len(raw_tensors), len(metadata)))

            # Metadata
            from parallax.gguf import _write_metadata_value, _write_string
            for k, v in metadata.items():
                _write_string(f, k)
                _write_metadata_value(f, v)

            # Tensors: quantize large 2D matrices to Q8_0, keep 1D norms as F32
            payload_chunks = []
            descriptors = []
            curr_offset = 0

            for name, arr in raw_tensors.items():
                pad = (alignment - (curr_offset % alignment)) % alignment
                curr_offset += pad
                if pad > 0:
                    payload_chunks.append(b"\x00" * pad)

                if arr.ndim == 2:
                    ggml_type = GGML_TYPE_Q8_0
                    raw_bytes = quantize_q8_0(arr)
                else:
                    ggml_type = 0  # F32
                    raw_bytes = arr.astype("<f4").tobytes()

                descriptors.append((name, arr.shape, ggml_type, curr_offset))
                payload_chunks.append(raw_bytes)
                curr_offset += len(raw_bytes)

            for name, shape, ggml_type, offset in descriptors:
                _write_string(f, name)
                f.write(struct.pack("<I", len(shape)))
                for dim in reversed(shape):
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<IQ", ggml_type, offset))

            # Header alignment
            hlen = f.tell()
            pad_h = (alignment - (hlen % alignment)) % alignment
            if pad_h > 0:
                f.write(b"\x00" * pad_h)

            for chunk in payload_chunks:
                f.write(chunk)

        # Load quantized model directly
        model = load_gguf_pretrained(tmp_path)
        assert isinstance(model, ParallaxLM)

        # Run forward pass inference
        idx = np.array([[1, 2, 3]], dtype=np.int64)
        logits = model(idx)
        assert logits.shape == (1, 3, vocab_size)
        assert not np.isnan(logits.data).any()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_quantize_extreme_magnitude_no_nan_overflow():
    """Verify high-magnitude float32 blocks quantize/dequantize without NaN."""
    # Values that exceed 127 * 65504 (Q8_0) and 7 * 65504 (Q4_0)
    extreme_val = 1e7
    arr = np.full((1, 32), extreme_val, dtype=np.float32)

    # Q8_0 check
    q8_bytes = quantize_q8_0(arr)
    deq8 = dequantize_q8_0(q8_bytes, arr.shape)
    assert not np.isnan(deq8).any()
    assert not np.isinf(deq8).any()

    # Q4_0 check
    q4_bytes = quantize_q4_0(arr)
    deq4 = dequantize_q4_0(q4_bytes, arr.shape)
    assert not np.isnan(deq4).any()
    assert not np.isinf(deq4).any()
