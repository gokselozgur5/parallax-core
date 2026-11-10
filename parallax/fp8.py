"""FP8 (E4M3 and E5M2) quantization and micro-scaling emulation."""

from typing import Tuple
import numpy as np


class FP8Format:
    """Defines floating-point bit layout and dynamic range limits."""

    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0


def _build_fp8_grid(exp_bits: int, mant_bits: int, max_val: float) -> np.ndarray:
    bias = (1 << (exp_bits - 1)) - 1
    vals = set()
    vals.add(0.0)
    for mant in range(1 << mant_bits):
        v = (mant / (1 << mant_bits)) * (2.0 ** (1 - bias))
        if v <= max_val:
            vals.add(v)
    max_exp = (1 << exp_bits) - 1
    for exp in range(1, max_exp + 1):
        for mant in range(1 << mant_bits):
            v = (1.0 + mant / (1 << mant_bits)) * (2.0 ** (exp - bias))
            if v <= max_val:
                vals.add(v)
    return np.array(sorted(vals), dtype=np.float64)


_E4M3_GRID = _build_fp8_grid(4, 3, FP8Format.E4M3_MAX)
_E5M2_GRID = _build_fp8_grid(5, 2, FP8Format.E5M2_MAX)


def _quantize_to_fp8_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Rounds values to nearest FP8 grid point with round-to-nearest-even tie-breaking."""
    abs_x = np.abs(x)
    signs = np.sign(x)
    idx = np.searchsorted(grid, abs_x, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    prev_idx = np.maximum(0, idx - 1)

    d_prev = abs_x - grid[prev_idx]
    d_curr = grid[idx] - abs_x
    choose_curr = d_curr < d_prev
    ties = np.isclose(d_curr, d_prev, atol=1e-12, rtol=1e-9)
    tie_even = (idx % 2 == 0)
    choose_curr = np.where(ties, tie_even, choose_curr)
    return signs * np.where(choose_curr, grid[idx], grid[prev_idx])


def quantize_fp8(
    tensor: np.ndarray,
    format_type: str = "e4m3",
    block_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize tensor to simulated FP8 representation with per-block scaling."""
    fmt = format_type.lower()
    if fmt in ("e4m3", "e4m3fn"):
        grid = _E4M3_GRID
        max_val = FP8Format.E4M3_MAX
    elif fmt == "e5m2":
        grid = _E5M2_GRID
        max_val = FP8Format.E5M2_MAX
    else:
        raise ValueError(
            f"Unsupported FP8 format_type: '{format_type}'. Supported: 'e4m3', 'e5m2'."
        )
    flat = tensor.flatten()
    pad_len = (block_size - (flat.size % block_size)) % block_size
    padded = np.pad(flat, (0, pad_len), mode="constant")
    blocks = padded.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / max_val
    scales = np.maximum(scales, 1e-12)
    quantized_blocks = _quantize_to_fp8_grid(blocks / scales, grid)
    return quantized_blocks, scales

def dequantize_fp8(
    quantized_blocks: np.ndarray,
    scales: np.ndarray,
    original_shape: Tuple[int, ...],
) -> np.ndarray:
    """Dequantize FP8 micro-scaled blocks back to original tensor shape."""
    reconstructed = (quantized_blocks * scales).flatten()
    total_elements = int(np.prod(original_shape))
    return reconstructed[:total_elements].reshape(original_shape)

def fp8_gemm(
    a: np.ndarray,
    b: np.ndarray,
    format_type: str = "e4m3",
    block_size: int = 16,
) -> np.ndarray:
    """Simulate FP8 GEMM with blockwise quantization and FP32 accumulation."""
    q_a, s_a = quantize_fp8(a, format_type=format_type, block_size=block_size)
    q_b, s_b = quantize_fp8(b, format_type=format_type, block_size=block_size)
    deq_a = dequantize_fp8(q_a, s_a, a.shape)
    deq_b = dequantize_fp8(q_b, s_b, b.shape)
    return np.matmul(deq_a, deq_b)
