"""
Unit tests for INT4 (4-bit) Quantization and Linear4Bit Operator.
"""

import numpy as np
import pytest

from parallax.int4 import (
    Linear4Bit,
    dequantize_int4_groupwise,
    pack_int4,
    quantize_int4_groupwise,
    unpack_int4,
)
from parallax.nn import Linear
from parallax.tensor import Tensor


def test_pack_and_unpack_int4():
    """Verify bitwise packing and unpacking preserves exact 4-bit integer values."""
    np.random.seed(42)
    # Generate random 4-bit integers in [0, 15]
    shape = (8, 64)
    original = np.random.randint(0, 16, size=shape, dtype=np.uint8)

    packed = pack_int4(original)
    assert packed.shape == (8, 32)
    assert packed.dtype == np.uint8

    unpacked = unpack_int4(packed)
    assert unpacked.shape == shape
    assert unpacked.dtype == np.uint8
    np.testing.assert_array_equal(original, unpacked)


def test_pack_int4_odd_dim_raises():
    """Packing with odd dimension should raise ValueError."""
    odd_arr = np.zeros((4, 15), dtype=np.uint8)
    with pytest.raises(ValueError, match="Last dimension must be even"):
        pack_int4(odd_arr)


def test_pack_int4_validation():
    """Verify pack_int4 rejects out of bounds or non-integer inputs."""
    with pytest.raises(ValueError, match="requires integer"):
        pack_int4(np.array([1.5, 2.0]))
    with pytest.raises(ValueError, match="must be in \\[0, 15\\]"):
        pack_int4(np.array([16, 0]))
    with pytest.raises(ValueError, match="must be in \\[0, 15\\]"):
        pack_int4(np.array([-1, 4]))


def test_quantize_float16_constant_group():
    """Verify float16 constant groups quantize with non-zero scale without warning."""
    w = np.full((4, 32), 0.5, dtype=np.float16)
    packed, scales, zeros = quantize_int4_groupwise(w, group_size=32)
    assert np.all(scales > 0)
    w_recon = dequantize_int4_groupwise(packed, scales, zeros, group_size=32)
    np.testing.assert_allclose(w_recon, w, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("group_size", [32, 128])
def test_quantize_dequantize_int4_groupwise_mse(group_size: int):
    """Verify group-wise quantization-dequantization error (MSE)."""
    np.random.seed(1337)
    out_features, in_features = 64, 256
    weight = np.random.randn(out_features, in_features).astype(np.float64)

    packed, scales, zeros = quantize_int4_groupwise(weight, group_size=group_size)

    # Check shapes
    assert packed.shape == (out_features, in_features // 2)
    assert scales.shape == (out_features, in_features // group_size)
    assert zeros.shape == (out_features, in_features // group_size)

    w_recon = dequantize_int4_groupwise(packed, scales, zeros, group_size=group_size)
    assert w_recon.shape == weight.shape

    # Mean squared error should be low for 4-bit quantization
    mse = np.mean((weight - w_recon) ** 2)
    # Relative variance / SNR check: normalized MSE = MSE / Var(w)
    nmse = mse / np.var(weight)
    assert nmse < 0.05  # 4-bit quantization typically achieves NMSE < 0.03


def test_compression_ratio():
    """Verify exact 4x / 8x memory compression in Linear4Bit."""
    np.random.seed(42)
    in_features, out_features = 256, 128
    lin = Linear(in_features, out_features, bias=False)

    l4 = Linear4Bit.from_linear(lin, group_size=32)
    stats = l4.memory_footprint()

    # Raw weight compression: 4 bytes (float32) vs 0.5 bytes (4-bit) = 8.0x
    # Against uint8 weight: 1 byte vs 0.5 bytes = 2.0x
    assert stats["weight_compression_ratio"] == 8.0
    assert stats["packed_ratio"] == 8.0

    # Total compression including scales and zeros:
    # 32 elements in float32 = 128 bytes
    # 32 elements in INT4 = 16 bytes + scale (4 bytes) + zero (4 bytes) = 24 bytes
    # Ratio = 128 / 24 = 5.33x
    assert stats["total_compression_ratio"] > 5.0
    assert stats["packed_bytes"] == (out_features * in_features) // 2

    # Baseline float64
    stats_f64 = l4.memory_footprint(baseline_dtype=np.float64)
    assert stats_f64["packed_ratio"] == 16.0
    assert stats_f64["total_compression_ratio"] > 10.0


def test_linear4bit_forward():
    """Verify forward pass output correctness and correlation with full float Linear."""
    np.random.seed(42)
    in_features, out_features = 128, 64
    lin = Linear(in_features, out_features, bias=True)
    l4 = Linear4Bit.from_linear(lin, group_size=32)

    x = Tensor(np.random.randn(4, 16, in_features))
    out_orig = lin(x)
    out_l4 = l4(x)

    assert out_l4.shape == (4, 16, out_features)

    # Cosine similarity / correlation between orig and int4 output should be very high (> 0.99)
    orig_flat = out_orig.data.flatten()
    l4_flat = out_l4.data.flatten()

    cos_sim = np.dot(orig_flat, l4_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(l4_flat))
    assert cos_sim > 0.99

    # Cache test
    l4._cached_weight = l4.dequantize()
    out_l4_cached = l4(x)
    np.testing.assert_array_equal(out_l4.data, out_l4_cached.data)
    l4.invalidate_cache()
    assert l4._cached_weight is None

    # Automatic caching test with cache_dequantized=True
    l4_cached = Linear4Bit.from_linear(lin, group_size=32, cache_dequantized=True)
    assert l4_cached._cached_weight is None
    call_count = 0
    orig_dequantize = l4_cached.dequantize

    def counted_dequantize():
        nonlocal call_count
        call_count += 1
        return orig_dequantize()

    l4_cached.dequantize = counted_dequantize
    _ = l4_cached(x)
    assert call_count == 1
    assert l4_cached._cached_weight is not None
    _ = l4_cached(x)
    assert call_count == 1


def test_linear4bit_gradient_propagation():
    """Verify gradients propagate through Linear4Bit to upstream layers."""
    np.random.seed(42)
    in_f, out_f = 32, 16
    pre = Linear(8, in_f, bias=True)
    lin = Linear(in_f, out_f, bias=True)
    q = Linear4Bit.from_linear(lin, group_size=32)

    x = Tensor(np.random.randn(2, 8), requires_grad=True)
    h = pre(x)
    y = q(h)
    loss = y.sum()
    loss.backward()

    assert h.grad is not None
    assert pre.weight.grad is not None
    assert x.grad is not None

    # Check analytical gradient against finite difference for pre.weight
    eps = 1e-6
    w_dequant = q.dequantize()
    w_arr = pre.weight.data.copy()
    num_grad_w = np.zeros_like(w_arr)
    for idx in np.ndindex(w_arr.shape):
        w_p = w_arr.copy()
        w_p[idx] += eps
        h_p = x.data @ w_p.T + pre.bias.data
        y_p = h_p @ w_dequant.T + q.bias_data
        loss_p = y_p.sum()

        w_m = w_arr.copy()
        w_m[idx] -= eps
        h_m = x.data @ w_m.T + pre.bias.data
        y_m = h_m @ w_dequant.T + q.bias_data
        loss_m = y_m.sum()
        num_grad_w[idx] = (loss_p - loss_m) / (2 * eps)

    np.testing.assert_allclose(pre.weight.grad, num_grad_w, rtol=1e-5, atol=1e-5)


def test_quantize_int4_exact_levels_kill_3bit():
    """Verify all 16 levels [0..15] are utilized; kills 3-bit mutation."""
    grid = np.tile(np.arange(16, dtype=np.float64), 2).reshape(1, 32)
    packed, scales, zeros = quantize_int4_groupwise(grid, group_size=32)
    unpacked = unpack_int4(packed)
    np.testing.assert_array_equal(unpacked, grid.astype(np.uint8))
    unique_levels = np.unique(unpacked)
    assert len(unique_levels) == 16
    assert unique_levels.max() == 15

