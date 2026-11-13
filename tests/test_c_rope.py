from __future__ import annotations

import numpy as np
import pytest
from parallax.native.rope import is_rope_available, rope_in_place


def test_rope_availability() -> None:
    assert is_rope_available()


def test_rope_invalid_shape_and_dtype() -> None:
    with pytest.raises(ValueError):
        rope_in_place(np.zeros((4, 8), dtype=np.float32))

    with pytest.raises(ValueError):
        rope_in_place(np.zeros((2, 4, 7), dtype=np.float32))

    with pytest.raises(TypeError):
        rope_in_place(np.zeros((2, 4, 8), dtype=np.float64))


def test_rope_zero_length() -> None:
    x = np.zeros((0, 4, 8), dtype=np.float32)
    out = rope_in_place(x)
    assert out.shape == (0, 4, 8)


def test_rope_parity_interleaved() -> None:
    from parallax.attention import apply_rope

    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, 4, 32)).astype(np.float32)
    x_t = np.transpose(x, (1, 0, 2))
    ref = apply_rope(x_t, seq_dim=1, offset=5, theta=10000.0, interleaved=True)
    expected = np.transpose(ref, (1, 0, 2))
    actual = rope_in_place(x.copy(), start_pos=5, theta=10000.0, interleaved=True)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_rope_parity_non_interleaved() -> None:
    from parallax.attention import apply_rope

    rng = np.random.default_rng(43)
    x = rng.standard_normal((6, 2, 16)).astype(np.float32)
    x_t = np.transpose(x, (1, 0, 2))
    ref = apply_rope(x_t, seq_dim=1, offset=0, theta=5000.0, interleaved=False)
    expected = np.transpose(ref, (1, 0, 2))
    actual = rope_in_place(x.copy(), start_pos=0, theta=5000.0, interleaved=False)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
