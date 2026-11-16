"""Unit tests for GQA head repeat dispatch helpers."""

import numpy as np
import pytest
from python.parallax.dispatch.gqa import repeat_kv


def test_repeat_kv_shape_and_values() -> None:
    x = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    # Shape: (1, 2, 1, 2) -> (batch=1, num_kv_heads=2, seq_len=1, head_dim=2)
    out = repeat_kv(x, n_rep=3)
    assert out.shape == (1, 6, 1, 2)
    np.testing.assert_allclose(out[:, 0:3, :, :], np.repeat(x[:, 0:1, :, :], 3, axis=1))
    np.testing.assert_allclose(out[:, 3:6, :, :], np.repeat(x[:, 1:2, :, :], 3, axis=1))


def test_repeat_kv_validation() -> None:
    x = np.ones((1, 2, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        repeat_kv(x, n_rep=0)
    with pytest.raises(ValueError):
        repeat_kv(x, n_rep=-2)
    with pytest.raises(ValueError):
        repeat_kv(np.ones((4, 8)), n_rep=2)


def test_reduce_kv_grad() -> None:
    from python.parallax.dispatch.gqa import reduce_kv_grad

    grad = np.ones((2, 6, 4, 8), dtype=np.float32)
    reduced = reduce_kv_grad(grad, n_rep=3)
    assert reduced.shape == (2, 2, 4, 8)
    np.testing.assert_allclose(reduced, np.full((2, 2, 4, 8), 3.0, dtype=np.float32))


