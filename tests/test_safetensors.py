"""
Unit tests for zero-dependency SafeTensors reader and writer.
"""

import json
import os
import struct
import tempfile
import numpy as np
import pytest

from parallax.safetensors import load_safetensors, save_safetensors


def test_safetensors_roundtrip_all_dtypes():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "model.safetensors")

        tensors = {
            "w_f64": np.array([[1.5, -2.25], [3.125, 4.0]], dtype=np.float64),
            "w_f32": np.array([0.1, -0.2, 0.3], dtype=np.float32),
            "w_f16": np.array([1.0, 2.0, -3.0], dtype=np.float16),
            "w_i64": np.array([1000000000, -2000000000], dtype=np.int64),
            "w_i32": np.array([100, -200, 300], dtype=np.int32),
            "w_i8": np.array([-128, 0, 127], dtype=np.int8),
            "w_u8": np.array([0, 128, 255], dtype=np.uint8),
            "w_bool": np.array([True, False, True], dtype=bool),
        }
        metadata = {"version": "1.0", "author": "Parallax"}

        save_safetensors(file_path, tensors, metadata=metadata)

        # 1. Load without cast to float64
        loaded_raw = load_safetensors(file_path, cast_float64=False)
        assert set(loaded_raw.keys()) == set(tensors.keys())

        for k, v in tensors.items():
            assert loaded_raw[k].dtype == v.dtype
            assert loaded_raw[k].shape == v.shape
            np.testing.assert_array_equal(loaded_raw[k], v)

        # 2. Load with cast to float64 (default)
        loaded_f64 = load_safetensors(file_path, cast_float64=True)
        assert loaded_f64["w_f32"].dtype == np.float64
        assert loaded_f64["w_f16"].dtype == np.float64
        np.testing.assert_allclose(loaded_f64["w_f32"], tensors["w_f32"], rtol=1e-6)
        np.testing.assert_allclose(loaded_f64["w_f16"], tensors["w_f16"], rtol=1e-3)


def test_safetensors_bf16_bitshift_decoding():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "bf16_model.safetensors")

        # Create known float32 values and truncate to BF16 (top 16 bits)
        orig_f32 = np.array([1.0, -2.0, 0.5, 3.140625], dtype=np.float32)
        u32_view = orig_f32.view(np.uint32)
        bf16_u16 = (u32_view >> 16).astype("<u2")
        bf16_bytes = bf16_u16.tobytes()

        # Write manual safetensors file containing BF16 tensor
        header = {
            "test_bf16": {
                "dtype": "BF16",
                "shape": [4],
                "data_offsets": [0, len(bf16_bytes)],
            }
        }
        header_json = json.dumps(header).encode("utf-8")
        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_json)))
            f.write(header_json)
            f.write(bf16_bytes)

        loaded = load_safetensors(file_path, cast_float64=True)
        assert "test_bf16" in loaded
        assert loaded["test_bf16"].dtype == np.float64
        np.testing.assert_allclose(loaded["test_bf16"], orig_f32, rtol=1e-2)


def test_safetensors_big_endian_conversion():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "be_model.safetensors")
        tensors = {
            "w_f32": np.array([1.5, -2.5, 3.75], dtype=">f4"),
            "w_f64": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=">f8"),
            "w_i32": np.array([10, -20, 30], dtype=">i4"),
        }
        save_safetensors(file_path, tensors)

        loaded = load_safetensors(file_path, cast_float64=False)
        for k, v in tensors.items():
            np.testing.assert_allclose(loaded[k], v)
            assert loaded[k].dtype.byteorder in ("<", "=", "|")


def test_safetensors_disjoint_offsets_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "overlap.safetensors")
        header = {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
        }
        h_bytes = json.dumps(header).encode("utf-8")
        with open(file_path, "wb") as f:
            f.write(struct.pack("<Q", len(h_bytes)))
            f.write(h_bytes)
            f.write(b"\x00" * 16)

        with pytest.raises(ValueError, match="Overlapping tensor offsets"):
            load_safetensors(file_path)
