"""Tests for Qwen3CUDAModel QK-Norm, custom head_dim, and streaming GGUF K-Quants."""

from pathlib import Path
import numpy as np
import pytest

from parallax.gguf import GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, save_gguf
from parallax.gguf_loader import load_gguf_pretrained
from parallax.gguf_quant import BLOCK_BYTES_Q4_K, BLOCK_BYTES_Q6_K, dequantize_q4_k, dequantize_q6_k


def test_torch_q4k_and_q6k_kernels_match_numpy() -> None:
    torch = pytest.importorskip("torch")
    from parallax.cuda_engine import _dequant_q4_k_torch, _dequant_q6_k_torch

    rng = np.random.default_rng(55)
    b4 = rng.integers(0, 256, size=(2, BLOCK_BYTES_Q4_K), dtype=np.uint8)
    b4[:, 0:4] = np.array([0x00, 0x3C, 0x00, 0x38], dtype=np.uint8)
    t4 = _dequant_q4_k_torch(torch.from_numpy(b4.reshape(-1)), (2, 256), torch.float32)
    np.testing.assert_allclose(t4.numpy(), dequantize_q4_k(b4.tobytes(), (2, 256)), atol=1e-6)

    b6 = rng.integers(0, 256, size=(2, BLOCK_BYTES_Q6_K), dtype=np.uint8)
    b6[:, 208:210] = np.array([0x00, 0x3C], dtype=np.uint8)
    t6 = _dequant_q6_k_torch(torch.from_numpy(b6.reshape(-1)), (2, 256), torch.float32)
    np.testing.assert_allclose(t6.numpy(), dequantize_q6_k(b6.tobytes(), (2, 256)), atol=1e-6)
