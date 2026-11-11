"""
Zero-dependency GGUF Quantized Tensors Dequantizer & Vectorized Kernels.

Implements high-throughput vectorized block dequantization for official GGML/GGUF
quantization formats (Q8_0 and Q4_0) using purely NumPy bitwise operations.
"""

from typing import Tuple
import numpy as np

# GGML Block Constants
QK8_0 = 32
BLOCK_BYTES_Q8_0 = 34  # 2 bytes FP16 delta + 32 bytes int8

QK4_0 = 32
BLOCK_BYTES_Q4_0 = 18  # 2 bytes FP16 delta + 16 bytes dual-nibble uint8


def dequantize_q8_0(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantizes a GGML Q8_0 byte buffer into an N-D float array."""
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // BLOCK_BYTES_Q8_0
    if n_blocks * BLOCK_BYTES_Q8_0 != len(raw):
        raise ValueError(
            f"Buffer length {len(raw)} is not a multiple of Q8_0 block size {BLOCK_BYTES_Q8_0}"
        )

    blocks = raw.reshape(n_blocks, BLOCK_BYTES_Q8_0)
    # Extract 2-byte FP16 scale factors
    deltas = blocks[:, :2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)
    # Extract 32-byte signed int8 values
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)           # (n_blocks, 32)

    dequant = deltas * qs
    return dequant.reshape(shape)


def quantize_q8_0(arr: np.ndarray) -> bytes:
    """Quantizes an N-D float array into GGML Q8_0 binary format."""
    flat = arr.astype(np.float32).flatten()
    n_elements = len(flat)
    if n_elements % QK8_0 != 0:
        raise ValueError(f"Array size {n_elements} must be a multiple of {QK8_0}")

    n_blocks = n_elements // QK8_0
    blocks = flat.reshape(n_blocks, QK8_0)

    # Compute max absolute value per block
    max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
    # Scale factor delta = max_abs / 127.0 clamped to FP16 max to prevent inf overflow
    fp16_max = np.finfo(np.float16).max
    clamped_scale = np.clip(max_abs / 127.0, 0.0, fp16_max)
    deltas = clamped_scale.astype("<f2")
    # Avoid division by zero
    scale = np.where(deltas.astype(np.float32) == 0.0, 1.0, deltas.astype(np.float32))

    # Quantize to int8: round(x / scale)
    qs = np.clip(np.round(blocks / scale), -128, 127).astype(np.int8)

    # Interleave deltas (2 bytes) and qs (32 bytes)
    out_bytes = bytearray()
    for b in range(n_blocks):
        out_bytes.extend(deltas[b].tobytes())
        out_bytes.extend(qs[b].tobytes())

    return bytes(out_bytes)


def dequantize_q4_0(raw_bytes: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantizes a GGML Q4_0 byte buffer into an N-D float array."""
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    n_blocks = len(raw) // BLOCK_BYTES_Q4_0
    if n_blocks * BLOCK_BYTES_Q4_0 != len(raw):
        raise ValueError(
            f"Buffer length {len(raw)} is not a multiple of Q4_0 block size {BLOCK_BYTES_Q4_0}"
        )

    blocks = raw.reshape(n_blocks, BLOCK_BYTES_Q4_0)
    # Extract 2-byte FP16 scale factors
    deltas = blocks[:, :2].copy().view("<f2").astype(np.float32)  # (n_blocks, 1)

    payload = blocks[:, 2:]  # (n_blocks, 16)
    # Low nibble: elements 0..15
    low = (payload & 0x0F).astype(np.int8) - 8                    # (n_blocks, 16)
    # High nibble: elements 16..31
    high = (payload >> 4).astype(np.int8) - 8                     # (n_blocks, 16)

    qs = np.concatenate([low, high], axis=1).astype(np.float32)   # (n_blocks, 32)
    dequant = deltas * qs
    return dequant.reshape(shape)


def quantize_q4_0(arr: np.ndarray) -> bytes:
    """Quantizes an N-D float array into GGML Q4_0 binary format."""
    flat = arr.astype(np.float32).flatten()
    n_elements = len(flat)
    if n_elements % QK4_0 != 0:
        raise ValueError(f"Array size {n_elements} must be a multiple of {QK4_0}")

    n_blocks = n_elements // QK4_0
    blocks = flat.reshape(n_blocks, QK4_0)

    # Compute signed extreme per block (matching ggml reference)
    abs_blocks = np.abs(blocks)
    max_abs_idx = np.argmax(abs_blocks, axis=1, keepdims=True)
    amax = np.take_along_axis(blocks, max_abs_idx, axis=1)

    raw_scale = np.where(amax < 0.0, amax / -8.0, amax / 7.0)
    fp16_max = np.finfo(np.float16).max
    clamped_scale = np.clip(raw_scale, 0.0, fp16_max)
    deltas = clamped_scale.astype("<f2")
    scale = np.where(deltas.astype(np.float32) == 0.0, 1.0, deltas.astype(np.float32))

    # Quantize: x / scale + 8
    qs = np.clip(np.round(blocks / scale) + 8, 0, 15).astype(np.uint8)

    # Pack into nibbles: low (0..15) and high (16..31)
    low = qs[:, :16]
    high = qs[:, 16:]
    packed = (low & 0x0F) | ((high & 0x0F) << 4)

    out_bytes = bytearray()
    for b in range(n_blocks):
        out_bytes.extend(deltas[b].tobytes())
        out_bytes.extend(packed[b].tobytes())

    return bytes(out_bytes)
