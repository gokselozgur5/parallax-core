"""FP8 (E4M3 and E5M2) quantization and micro-scaling emulation."""

from typing import Tuple
import numpy as np


class FP8Format:
    """Defines floating-point bit layout and dynamic range limits."""

    E4M3_MAX = 448.0
    E5M2_MAX = 57344.0

def quantize_fp8(
    tensor: np.ndarray,
    format_type: str = "e4m3",
    block_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize tensor to simulated FP8 representation with per-block scaling."""
    max_val = FP8Format.E4M3_MAX if format_type == "e4m3" else FP8Format.E5M2_MAX
    flat = tensor.flatten()
    pad_len = (block_size - (flat.size % block_size)) % block_size
    padded = np.pad(flat, (0, pad_len), mode="constant")
    blocks = padded.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / max_val
    scales = np.maximum(scales, 1e-12)
    quantized_blocks = np.clip(np.round(blocks / scales), -max_val, max_val)
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
