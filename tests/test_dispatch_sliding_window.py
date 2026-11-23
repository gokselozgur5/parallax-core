"""Unit tests for sliding window FlashAttention dispatch helpers."""

import numpy as np
import pytest
from parallax.dispatch.sliding_window import build_sliding_window_mask


def test_build_sliding_window_mask_causal() -> None:
    mask = build_sliding_window_mask(seq_len_q=4, seq_len_k=4, window_size=1, causal=True)
    expected = np.array([
        [True, False, False, False],
        [True, True, False, False],
        [False, True, True, False],
        [False, False, True, True],
    ])
    np.testing.assert_array_equal(mask, expected)


def test_get_sliding_window_bounds() -> None:
    from parallax.dispatch.sliding_window import get_sliding_window_bounds

    starts, ends = get_sliding_window_bounds(seq_len_q=4, seq_len_k=4, window_size=2, causal=True)
    np.testing.assert_array_equal(starts, np.array([0, 0, 0, 1]))
    np.testing.assert_array_equal(ends, np.array([1, 2, 3, 4]))

    starts_nc, ends_nc = get_sliding_window_bounds(seq_len_q=4, seq_len_k=4, window_size=1, causal=False)
    np.testing.assert_array_equal(starts_nc, np.array([0, 0, 1, 2]))
    np.testing.assert_array_equal(ends_nc, np.array([2, 3, 4, 4]))


def test_sliding_window_attention_shapes_and_values() -> None:
    from parallax.dispatch.sliding_window import sliding_window_attention

    rng = np.random.default_rng(42)
    q = rng.standard_normal((4, 2, 8)).astype(np.float32)
    k = rng.standard_normal((4, 2, 8)).astype(np.float32)
    v = rng.standard_normal((4, 2, 8)).astype(np.float32)
    out = sliding_window_attention(q, k, v, window_size=1, causal=True)
    assert out.shape == q.shape
    assert out.dtype == np.float32


def test_sliding_window_zero_window_identity() -> None:
    from parallax.dispatch.sliding_window import sliding_window_attention

    q = np.ones((3, 1, 4), dtype=np.float32)
    k = np.ones((3, 1, 4), dtype=np.float32)
    v = np.array([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]], [[9.0, 10.0, 11.0, 12.0]]], dtype=np.float32)
    out = sliding_window_attention(q, k, v, window_size=0, causal=True)
    np.testing.assert_allclose(out, v, atol=1e-5)


def test_flash_sliding_window_tile() -> None:
    from parallax.dispatch.sliding_window import flash_sliding_window_tile

    q_block = np.ones((2, 4), dtype=np.float32)
    k_block = np.ones((2, 4), dtype=np.float32)
    v_block = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out, m, l = flash_sliding_window_tile(
        q_block, k_block, v_block, q_start=0, k_start=0, window_size=1, scale=0.5
    )
    assert out.shape == (2, 2)
    assert m.shape == (2, 1)
    assert l.shape == (2, 1)


def test_sliding_window_attention_class() -> None:
    from parallax.dispatch.sliding_window import SlidingWindowAttention

    layer = SlidingWindowAttention(window_size=2, causal=True)
    assert layer.window_size == 2
    assert layer.causal is True
    q = np.ones((4, 2, 8), dtype=np.float32)
    k = np.ones((4, 2, 8), dtype=np.float32)
    v = np.ones((4, 2, 8), dtype=np.float32)
    out = layer(q, k, v)
    assert out.shape == q.shape


def test_dispatched_sliding_window_attention() -> None:
    from parallax.dispatch.sliding_window import dispatched_sliding_window_attention

    q = np.ones((2, 1, 4), dtype=np.float32)
    k = np.ones((2, 1, 4), dtype=np.float32)
    v = np.ones((2, 1, 4), dtype=np.float32)
    out = dispatched_sliding_window_attention(q, k, v, window_size=1)
    assert out.shape == q.shape


def test_package_exports() -> None:
    import parallax.dispatch as pd
    import parallax.dispatch as ppd

    for mod in (pd, ppd):
        assert hasattr(mod, "SlidingWindowAttention")
        assert hasattr(mod, "build_sliding_window_mask")
        assert hasattr(mod, "dispatched_sliding_window_attention")
        assert hasattr(mod, "flash_sliding_window_tile")
        assert hasattr(mod, "get_sliding_window_bounds")
        assert hasattr(mod, "sliding_window_attention")


def test_sliding_window_validation() -> None:
    from parallax.dispatch.sliding_window import build_sliding_window_mask, SlidingWindowAttention

    with pytest.raises(ValueError):
        build_sliding_window_mask(seq_len_q=4, seq_len_k=4, window_size=-1)
    with pytest.raises(ValueError):
        SlidingWindowAttention(window_size=-2)
