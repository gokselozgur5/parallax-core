"""Unit tests for Multi-Query Attention (MQA) head broadcast helpers."""

import numpy as np
import pytest

from parallax.dispatch.mqa import broadcast_kv, reduce_mqa_grad


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


def test_dispatched_broadcast_kv_autograd() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.mqa import dispatched_broadcast_kv

    x = Tensor(np.ones((1, 1, 3, 4), dtype=np.float32), requires_grad=True)
    out = dispatched_broadcast_kv(x, num_heads=4)
    assert out.shape == (1, 4, 3, 4)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 1, 3, 4)
    np.testing.assert_allclose(x.grad, np.full((1, 1, 3, 4), 4.0, dtype=np.float32))

    x3 = Tensor(np.ones((2, 3, 4), dtype=np.float32), requires_grad=True)
    out3 = dispatched_broadcast_kv(x3, num_heads=3)
    assert out3.shape == (2, 3, 3, 4)
    out3.sum().backward()
    assert x3.grad is not None
    assert x3.grad.shape == (2, 3, 4)
    np.testing.assert_allclose(x3.grad, np.full((2, 3, 4), 3.0, dtype=np.float32))


def test_mqa_head_broadcaster_module() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.mqa import MQAHeadBroadcaster

    broadcaster = MQAHeadBroadcaster(num_heads=8)
    assert broadcaster.num_heads == 8
    assert "num_heads=8" in broadcaster.extra_repr()
    x = Tensor(np.random.randn(2, 1, 8, 16).astype(np.float32))
    out = broadcaster(x)
    assert out.shape == (2, 8, 8, 16)
    with pytest.raises(ValueError):
        MQAHeadBroadcaster(num_heads=0)


def test_broadcast_mqa_pair() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.mqa import broadcast_mqa_pair

    k = Tensor(np.ones((1, 1, 4, 8), dtype=np.float32))
    v = Tensor(np.full((1, 1, 4, 8), 2.0, dtype=np.float32))
    k_exp, v_exp = broadcast_mqa_pair(k, v, num_heads=4)
    assert k_exp.shape == (1, 4, 4, 8)
    assert v_exp.shape == (1, 4, 4, 8)


def test_dispatch_mqa_package_exports() -> None:
    from parallax.dispatch import mqa

    assert hasattr(mqa, "broadcast_kv")
    assert hasattr(mqa, "reduce_mqa_grad")
    assert hasattr(mqa, "dispatched_broadcast_kv")
    assert hasattr(mqa, "MQAHeadBroadcaster")
    assert hasattr(mqa, "broadcast_mqa_pair")


def test_parallax_dispatch_mqa_export() -> None:
    from parallax.dispatch import MQAHeadBroadcaster, broadcast_kv, dispatched_broadcast_kv

    assert MQAHeadBroadcaster is not None
    assert broadcast_kv is not None
    assert dispatched_broadcast_kv is not None
