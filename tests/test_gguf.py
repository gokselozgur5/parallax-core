import os
import struct
import tempfile
import numpy as np
import pytest

from parallax.gguf import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFFile,
    load_gguf,
    save_gguf,
)


def test_gguf_save_and_load_roundtrip():
    metadata = {
        "general.architecture": "llama",
        "general.name": "TestModel",
        "llama.context_length": 4096,
        "llama.embedding_length": 128,
        "llama.attention.head_count": 4,
        "llama.rope.freq_base": 10000.0,
        "general.file_type": 1,
        "tokenizer.ggml.tokens": ["<pad>", "hello", "world"],
        "tokenizer.ggml.scores": [0.0, -1.5, -2.0],
        "test.flag": True,
    }

    tensors = {
        "blk.0.attn_q.weight": np.random.randn(32, 128).astype(np.float32),
        "blk.0.attn_k.weight": np.random.randn(32, 128).astype(np.float16),
        "token_embd.weight": np.random.randn(100, 128).astype(np.float64),
    }

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_gguf(tmp_path, metadata=metadata, tensors=tensors, version=3)
        gguf_file = load_gguf(tmp_path, cast_float64=True)

        assert gguf_file.version == 3
        assert gguf_file.metadata["general.architecture"] == "llama"
        assert gguf_file.metadata["general.name"] == "TestModel"
        assert gguf_file.metadata["llama.context_length"] == 4096
        assert gguf_file.metadata["test.flag"] is True
        assert gguf_file.metadata["tokenizer.ggml.tokens"] == ["<pad>", "hello", "world"]

        # Check tensors
        assert "blk.0.attn_q.weight" in gguf_file.tensors
        assert "blk.0.attn_k.weight" in gguf_file.tensors
        assert "token_embd.weight" in gguf_file.tensors

        # Shapes must match row-major dimensions
        assert gguf_file.tensors["blk.0.attn_q.weight"].shape == (32, 128)
        assert gguf_file.tensors["blk.0.attn_k.weight"].shape == (32, 128)
        assert gguf_file.tensors["token_embd.weight"].shape == (100, 128)

        # Values match within float precision
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.attn_q.weight"],
            tensors["blk.0.attn_q.weight"],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            gguf_file.tensors["blk.0.attn_k.weight"],
            tensors["blk.0.attn_k.weight"].astype(np.float32),
            rtol=1e-3,
            atol=1e-3,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_bf16_parsing():
    # Construct a minimal synthetic GGUF v3 file containing BF16 data
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        # Expected float values: [1.0, -2.0, 0.5, 0.0]
        # In BF16, float32 bits shifted right by 16:
        f32_vals = np.array([1.0, -2.0, 0.5, 0.0], dtype=np.float32)
        u32_vals = f32_vals.view(np.uint32)
        bf16_u16 = (u32_vals >> 16).astype("<u2")
        raw_bf16_bytes = bf16_u16.tobytes()

        alignment = 32
        with open(tmp_path, "wb") as f:
            # Header: magic (4), version (4), tensor_count (8), metadata_count (8)
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<IQQ", 3, 1, 0))

            # Tensor descriptor: name, n_dims, dims (inner-to-outer), type, offset
            name_bytes = "test_bf16".encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", 1))  # 1 dim
            f.write(struct.pack("<Q", 4))  # len 4
            f.write(struct.pack("<IQ", GGML_TYPE_BF16, 0))  # offset 0

            # Pad header to alignment boundary
            header_len = f.tell()
            pad = (alignment - (header_len % alignment)) % alignment
            if pad > 0:
                f.write(b"\x00" * pad)

            # Write tensor data
            f.write(raw_bf16_bytes)

        gguf_file = load_gguf(tmp_path, cast_float64=False)
        assert "test_bf16" in gguf_file.tensors
        parsed = gguf_file.tensors["test_bf16"]
        assert parsed.shape == (4,)
        np.testing.assert_allclose(parsed, f32_vals, rtol=1e-5, atol=1e-5)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_invalid_magic_and_version():
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name
        f.write(b"BADM\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")

    try:
        with pytest.raises(ValueError, match="Invalid GGUF magic"):
            load_gguf(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_bool_and_int64_array_serialization():
    metadata = {
        "flags": [True, False, True],
        "wide_ints": [2**40, -(2**40)],
        "normal_ints": [1, 2, 3],
    }
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name

    try:
        save_gguf(tmp_path, metadata=metadata, tensors={})
        loaded = load_gguf(tmp_path)
        assert loaded.metadata["flags"] == [True, False, True]
        assert loaded.metadata["wide_ints"] == [2**40, -(2**40)]
        assert loaded.metadata["normal_ints"] == [1, 2, 3]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gguf_nested_array_depth_limit():
    # Build a synthetic GGUF file with deeply nested arrays exceeding depth limit
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        tmp_path = f.name
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, 0, 1))  # 0 tensors, 1 metadata kv
        key = "nested".encode("utf-8")
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        f.write(struct.pack("<I", 9))  # GGUF_TYPE_ARRAY
        # Nest 20 levels deep (MAX_GGUF_ARRAY_DEPTH is 16)
        for _ in range(20):
            f.write(struct.pack("<IQ", 9, 1))  # GGUF_TYPE_ARRAY, count 1
        f.write(struct.pack("<IQ", 5, 1))  # GGUF_TYPE_INT32, count 1
        f.write(struct.pack("<i", 42))

    try:
        with pytest.raises(ValueError, match="GGUF array nesting depth exceeds maximum limit"):
            load_gguf(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
