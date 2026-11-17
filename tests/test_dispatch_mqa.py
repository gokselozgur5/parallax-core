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
