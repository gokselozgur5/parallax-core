"""Unit tests for Multi-Query Attention (MQA) head broadcast helpers."""

import numpy as np
import pytest

from python.parallax.dispatch.mqa import broadcast_kv, reduce_mqa_grad


def test_broadcast_kv_4d() -> None:
    x = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)
    with pytest.raises(ValueError):
        broadcast_kv(x, num_heads=4)

    x_single = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    out = broadcast_kv(x_single, num_heads=4)
    assert out.shape == (1, 4, 2, 2)
    for h in range(4):
        np.testing.assert_allclose(out[:, h : h + 1, :, :], x_single)


def test_broadcast_kv_3d() -> None:
    x = np.ones((2, 5, 8), dtype=np.float32)
    out = broadcast_kv(x, num_heads=4)
    assert out.shape == (2, 4, 5, 8)
    np.testing.assert_allclose(out[:, 0, :, :], x)


def test_broadcast_kv_validation() -> None:
    x = np.ones((1, 1, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        broadcast_kv(x, num_heads=0)
    with pytest.raises(ValueError):
        broadcast_kv(x, num_heads=-1)
    with pytest.raises(ValueError):
        broadcast_kv(np.ones((4, 8)), num_heads=2)


def test_reduce_mqa_grad() -> None:
    grad = np.ones((2, 4, 3, 6), dtype=np.float32)
    red4 = reduce_mqa_grad(grad, original_ndim=4)
    assert red4.shape == (2, 1, 3, 6)
    np.testing.assert_allclose(red4, np.full((2, 1, 3, 6), 4.0, dtype=np.float32))

    red3 = reduce_mqa_grad(grad, original_ndim=3)
    assert red3.shape == (2, 3, 6)
    np.testing.assert_allclose(red3, np.full((2, 3, 6), 4.0, dtype=np.float32))

    with pytest.raises(ValueError):
        reduce_mqa_grad(grad, original_ndim=2)
    with pytest.raises(ValueError):
        reduce_mqa_grad(np.ones((4, 3, 6), dtype=np.float32))
