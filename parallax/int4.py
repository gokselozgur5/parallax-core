"""
INT4 (4-bit) Weight Quantization Primitives and Layers in Pure NumPy.

Implements:
- 4-bit nibble packing and unpacking (`pack_int4`, `unpack_int4`).
- Group-wise affine (scale and zero-point) quantization (`quantize_int4_groupwise`, `dequantize_int4_groupwise`).
- `Linear4Bit` inference module: stores weights packed into uint8 (2x compression over uint8, 4x over float16, 8x over float32 / 16x over float64),
  with on-the-fly or cached dequantization in the forward pass.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from parallax.nn import Linear, Module
from parallax.tensor import Tensor


def pack_int4(arr: np.ndarray) -> np.ndarray:
    """Packs an array of unsigned 4-bit integers [0, 15] into uint8.

    Pairs adjacent elements along the last axis:
      packed = (low_nibble & 0x0F) | ((high_nibble & 0x0F) << 4)

    Args:
        arr: Array of integer values in [0, 15]. The last dimension must be even.

    Returns:
        Array of dtype uint8 with last dimension halved (shape[:-1] + (shape[-1] // 2,)).
    """
    if arr.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {arr.shape[-1]}")

    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)):
        raise ValueError(f"pack_int4 requires integer array, got {arr.dtype}")
    if np.any((arr < 0) | (arr > 15)):
        raise ValueError("pack_int4 values must be in [0, 15]")

    arr_u8 = arr.astype(np.uint8) & 0x0F
    low = arr_u8[..., 0::2]
    high = arr_u8[..., 1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return packed


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Unpacks a uint8 array of packed 4-bit integers into separate uint8 elements [0, 15].

    Args:
        packed: uint8 array with packed nibbles.

    Returns:
        uint8 array with last dimension doubled (shape[:-1] + (shape[-1] * 2,)).
    """
    low = (packed & 0x0F).astype(np.uint8)
    high = ((packed >> 4) & 0x0F).astype(np.uint8)

    # Interleave low and high along last axis
    out_shape = list(packed.shape)
    out_shape[-1] *= 2

    unpacked = np.empty(out_shape, dtype=np.uint8)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high
    return unpacked


def quantize_int4_groupwise(
    weight: np.ndarray,
    group_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantizes a 2D weight matrix [out_features, in_features] to 4-bit unsigned [0, 15]

    using group-wise affine quantization:
        w_quant = round((w - min_val) / scale)
        w_quant = clip(w_quant, 0, 15)

    where:
        scale = (max_val - min_val) / 15.0
        zero_point = min_val

    Args:
        weight: 2D array [out_features, in_features] of floats.
        group_size: Number of consecutive input elements per quantization group.
                    Must divide in_features evenly.

    Returns:
        packed_weight: uint8 array of shape [out_features, in_features // 2]
        scales: float32/float64 array of shape [out_features, num_groups]
        zeros: float32/float64 array of shape [out_features, num_groups]
    """
    if weight.ndim != 2:
        raise ValueError(f"Weight must be a 2D array, got {weight.ndim}D")

    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

    num_groups = in_features // group_size
    w_grouped = weight.reshape(out_features, num_groups, group_size)

    w_min = np.min(w_grouped, axis=2, keepdims=True)
    w_max = np.max(w_grouped, axis=2, keepdims=True)

    floor = (
        np.finfo(weight.dtype).tiny
        if np.issubdtype(weight.dtype, np.floating)
        else 1e-8
    )
    denom = np.maximum(w_max - w_min, floor)
    scales = denom / 15.0
    zeros = w_min

    q = np.round((w_grouped - zeros) / scales)
    q = np.clip(q, 0, 15).astype(np.uint8)

    # Flatten back to [out_features, in_features] before packing
    q_flat = q.reshape(out_features, in_features)
    packed_weight = pack_int4(q_flat)

    # Squeeze group dimension from scales and zeros: [out_features, num_groups]
    return packed_weight, scales.squeeze(axis=2), zeros.squeeze(axis=2)


def dequantize_int4_groupwise(
    packed_weight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = 32,
) -> np.ndarray:
    """Dequantizes a packed 4-bit weight matrix back to float.

    w = q * scale + zero

    Args:
        packed_weight: uint8 array of shape [out_features, in_features // 2].
        scales: float array of shape [out_features, num_groups].
        zeros: float array of shape [out_features, num_groups].
        group_size: Number of elements per group.

    Returns:
        Dequantized weight matrix of shape [out_features, in_features] in float64.
    """
    out_features = packed_weight.shape[0]
    unpacked = unpack_int4(packed_weight)  # [out_features, in_features]
    in_features = unpacked.shape[1]
    num_groups = in_features // group_size

    unpacked_grouped = unpacked.reshape(out_features, num_groups, group_size).astype(np.float64)
    s = scales[:, :, np.newaxis].astype(np.float64)
    z = zeros[:, :, np.newaxis].astype(np.float64)

    w_recon = unpacked_grouped * s + z
    return w_recon.reshape(out_features, in_features)


class Linear4Bit(Module):
    """Linear projection layer storing weights in 4-bit packed format (uint8 nibbles).

    Performs group-wise dequantization on-the-fly or uses cached weights for repeated forward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 32,
    ):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )
        if in_features % 2 != 0:
            raise ValueError(f"in_features ({in_features}) must be even for nibble packing")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size

        # Packed weights: uint8 array of shape [out_features, in_features // 2]
        self.packed_weight: np.ndarray = np.zeros(
            (out_features, in_features // 2), dtype=np.uint8
        )
        self.scales: np.ndarray = np.ones((out_features, self.num_groups), dtype=np.float32)
        self.zeros: np.ndarray = np.zeros((out_features, self.num_groups), dtype=np.float32)
        self.bias_data: Optional[np.ndarray] = (
            np.zeros(out_features, dtype=np.float64) if bias else None
        )
        self._cached_weight: Optional[np.ndarray] = None

    def invalidate_cache(self) -> None:
        """Invalidates cached dequantized weights."""
        self._cached_weight = None

    @classmethod
    def from_linear(
        cls,
        linear: Linear,
        group_size: int = 32,
    ) -> "Linear4Bit":
        """Constructs a Linear4Bit instance from a trained float Linear layer."""
        l4 = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            group_size=group_size,
        )
        packed, scales, zeros = quantize_int4_groupwise(
            linear.weight.data, group_size=group_size
        )
        l4.packed_weight = packed
        l4.scales = scales.astype(np.float32)
        l4.zeros = zeros.astype(np.float32)
        if linear.bias is not None:
            l4.bias_data = linear.bias.data.copy()
        return l4

    def dequantize(self) -> np.ndarray:
        """Returns the dequantized full-precision weight matrix [out_features, in_features]."""
        return dequantize_int4_groupwise(
            self.packed_weight,
            self.scales,
            self.zeros,
            group_size=self.group_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with on-the-fly or cached dequantized weight projection.

        Args:
            x: Input Tensor of shape [..., in_features]

        Returns:
            Output Tensor of shape [..., out_features]
        """
        if self._cached_weight is None:
            w_float = self.dequantize()
        else:
            w_float = self._cached_weight

        out_data = x.data @ w_float.T
        if self.bias_data is not None:
            out_data = out_data + self.bias_data

        return Tensor(out_data, (x,), "Linear4Bit")

    def memory_footprint(self) -> Dict[str, Union[int, float]]:
        """Returns memory breakdown in bytes and compression stats against float32 baseline."""
        packed_bytes = self.packed_weight.nbytes
        scale_bytes = self.scales.nbytes
        zero_bytes = self.zeros.nbytes
        bias_bytes = self.bias_data.nbytes if self.bias_data is not None else 0
        total_quant_bytes = packed_bytes + scale_bytes + zero_bytes + bias_bytes

        # Baseline: float32 weight + bias
        weight_elements = self.out_features * self.in_features
        bias_elements = self.out_features if self.bias_data is not None else 0
        baseline_bytes = (weight_elements + bias_elements) * 4

        # Weight-only compression (raw packing ratio)
        weight_compression = (weight_elements * 4) / packed_bytes
        total_compression = baseline_bytes / total_quant_bytes

        return {
            "packed_bytes": packed_bytes,
            "scale_bytes": scale_bytes,
            "zero_bytes": zero_bytes,
            "bias_bytes": bias_bytes,
            "total_quant_bytes": total_quant_bytes,
            "baseline_float32_bytes": baseline_bytes,
            "weight_compression_ratio": weight_compression,
            "total_compression_ratio": total_compression,
        }
