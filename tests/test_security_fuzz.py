"""
Red Team Security Fuzzing Tests for SafeTensors and GGUF Loaders.

Tests adversarial inputs including:
- Extremely large headers (>100MB) causing Denial of Service (DoS)
- Integer overflows, negative offsets, and negative dimensions
- Out-of-bounds byte ranges beyond file boundaries
- Malformed JSON headers, truncated magic headers, and invalid metadata
Ensures graceful rejection via ValueError without crashing or segfaulting.
"""

import json
import math
import os
import struct
import tempfile
import numpy as np
import pytest

from parallax.safetensors import load_safetensors, MAX_HEADER_SIZE
from parallax.gguf import (
    load_gguf,
    GGUF_MAGIC,
    GGML_TYPE_F32,
    MAX_GGUF_HEADER_SIZE,
    MAX_GGUF_STRING_LEN,
    MAX_GGUF_ARRAY_LEN,
)


# =====================================================================
# SafeTensors Red Team Security Fuzzing Tests
# =====================================================================


def test_safetensors_dos_oversized_header():
    """Header size indicator claims >100MB to trigger DoS allocation."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        # Claim header is 200MB (exceeds MAX_HEADER_SIZE 100MB)
        f.write(struct.pack("<Q", 200 * 1024 * 1024))
        f.write(b"{}")

    try:
        with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_truncated_file_less_than_8_bytes():
    """File has fewer than 8 bytes."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(b"TINY")

    try:
        with pytest.raises(ValueError, match="smaller than 8 bytes"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_truncated_header_payload():
    """Header length claims 500 bytes but file only has 10 bytes."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(struct.pack("<Q", 500))
        f.write(b"short")

    try:
        with pytest.raises(ValueError, match="Truncated header in safetensors file"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_malformed_json_header():
    """Header contains corrupt/malformed non-JSON data."""
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        bad_json = b"{'key': unquoted_garbage!@#$}"
        f.write(struct.pack("<Q", len(bad_json)))
        f.write(bad_json)

    try:
        with pytest.raises(ValueError, match="Malformed JSON header"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_negative_offsets():
    """Negative data offsets attempting memory underflow."""
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [-10, 6],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 32)

    try:
        with pytest.raises(ValueError, match="Invalid tensor offsets|Invalid tensor offset/length"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_inverted_offsets():
    """Start offset greater than end offset (negative length)."""
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [20, 10],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 32)

    try:
        with pytest.raises(ValueError, match="Invalid tensor offsets"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_out_of_bounds_offset():
    """Offsets pointing past end-of-file (out-of-bounds read)."""
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [4],
            "data_offsets": [0, 1000],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 16)  # Only 16 bytes provided

    try:
        with pytest.raises(ValueError, match="Invalid tensor offset/length: exceeds file bounds"):
            load_safetensors(path)
    finally:
        os.remove(path)


def test_safetensors_negative_shape():
    """Negative shape dimensions."""
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [-4, 2],
            "data_offsets": [0, 16],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * 16)

    try:
        with pytest.raises(ValueError, match="Invalid dimension"):
            load_safetensors(path)
    finally:
        os.remove(path)


# =====================================================================
# GGUF Red Team Security Fuzzing Tests
# =====================================================================


def test_gguf_corrupted_magic():
    """Corrupted magic bytes."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(b"BADM\x03\x00\x00\x00")

    try:
        with pytest.raises(ValueError, match="Invalid GGUF magic header"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_unsupported_version():
    """GGUF version that is unsupported (e.g. v99 or v1)."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<I", 99))

    try:
        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_dos_excessive_counts():
    """Header claiming 50,000,000 tensors to exhaust memory/CPU."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, 50_000_000, 0))

    try:
        with pytest.raises(ValueError, match="Excessive GGUF counts"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_dos_oversized_string():
    """String metadata claiming 500MB string length."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        # 0 tensors, 1 metadata entry
        f.write(struct.pack("<IQQ", 3, 0, 1))
        # Write string length of 500MB
        f.write(struct.pack("<Q", 500 * 1024 * 1024))
        f.write(b"test")

    try:
        with pytest.raises(ValueError, match="GGUF string length .* exceeds maximum allowed limit"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_dos_oversized_array_length():
    """Array metadata claiming 100,000,000 elements."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, 0, 1))
        # Key name
        key = b"huge_arr"
        f.write(struct.pack("<Q", len(key)))
        f.write(key)
        # Type ARRAY (9), inner type UINT32 (4), count 100M
        f.write(struct.pack("<IIQ", 9, 4, 100_000_000))

    try:
        with pytest.raises(ValueError, match="GGUF array element count .* exceeds maximum limit"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_out_of_bounds_tensor_offset():
    """Tensor descriptor offset points past end of file."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        # 1 tensor, 0 metadata
        f.write(struct.pack("<IQQ", 3, 1, 0))
        # Tensor descriptor: name="w", n_dims=1, dim=[4], type=F32, offset=999999
        name = b"w"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", 4))
        f.write(struct.pack("<IQ", GGML_TYPE_F32, 999999))
        # Alignment padding (alignment=32)
        cur = f.tell()
        pad = (32 - (cur % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * 16)  # Data payload is only 16 bytes

    try:
        with pytest.raises(ValueError, match="Invalid tensor offset/length: exceeds file bounds"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_dimension_integer_overflow():
    """Tensor dimensions whose product overflows 64-bit bounds or produces absurd size."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        # 1 tensor, 0 metadata
        f.write(struct.pack("<IQQ", 3, 1, 0))
        name = b"overflow_tensor"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        # 2 dimensions: 2^40 and 2^40 -> product exceeds address space
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<QQ", 1 << 40, 1 << 40))
        f.write(struct.pack("<IQ", GGML_TYPE_F32, 0))
        cur = f.tell()
        pad = (32 - (cur % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * 16)

    try:
        with pytest.raises(ValueError, match="exceeds file bounds|overflow"):
            load_gguf(path)
    finally:
        os.remove(path)


def test_gguf_truncated_tensor_data():
    """Tensor specifies shape requiring 16 bytes, but file ends prematurely."""
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        path = f.name
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", 3, 1, 0))
        name = b"trunc_w"
        f.write(struct.pack("<Q", len(name)))
        f.write(name)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", 4))  # 4 * 4 = 16 bytes
        f.write(struct.pack("<IQ", GGML_TYPE_F32, 0))
        cur = f.tell()
        pad = (32 - (cur % 32)) % 32
        f.write(b"\x00" * pad)
        f.write(b"\x00" * 4)  # Only 4 bytes written instead of 16

    try:
        with pytest.raises(ValueError, match="Invalid tensor offset/length: exceeds file bounds"):
            load_gguf(path)
    finally:
        os.remove(path)
