"""Unit tests for GQA head repeat dispatch helpers."""

import numpy as np
import pytest
from parallax.dispatch.gqa import repeat_kv


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
    from parallax.dispatch.gqa import reduce_kv_grad

    grad = np.ones((2, 6, 4, 8), dtype=np.float32)
    reduced = reduce_kv_grad(grad, n_rep=3)
    assert reduced.shape == (2, 2, 4, 8)
    np.testing.assert_allclose(reduced, np.full((2, 2, 4, 8), 3.0, dtype=np.float32))


def test_dispatched_repeat_kv_autograd() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.gqa import dispatched_repeat_kv

    x = Tensor(np.ones((1, 2, 3, 4), dtype=np.float32), requires_grad=True)
    out = dispatched_repeat_kv(x, n_rep=3)
    assert out.shape == (1, 6, 3, 4)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 2, 3, 4)
    np.testing.assert_allclose(x.grad, np.full((1, 2, 3, 4), 3.0, dtype=np.float32))


def test_gqa_head_repeater_module() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.gqa import GQAHeadRepeater

    repeater = GQAHeadRepeater(n_rep=4)
    assert repeater.n_rep == 4
    assert "n_rep=4" in repeater.extra_repr()
    x = Tensor(np.random.randn(2, 2, 8, 16).astype(np.float32))
    out = repeater(x)
    assert out.shape == (2, 8, 8, 16)


def test_repeat_kv_pair() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.gqa import repeat_kv_pair

    k = Tensor(np.ones((1, 2, 4, 8), dtype=np.float32))
    v = Tensor(np.full((1, 2, 4, 8), 2.0, dtype=np.float32))
    k_exp, v_exp = repeat_kv_pair(k, v, n_rep=2)
    assert k_exp.shape == (1, 4, 4, 8)
    assert v_exp.shape == (1, 4, 4, 8)


def test_dispatch_gqa_package_exports() -> None:
    from parallax.dispatch import gqa

    assert hasattr(gqa, "repeat_kv")
    assert hasattr(gqa, "reduce_kv_grad")
    assert hasattr(gqa, "dispatched_repeat_kv")
    assert hasattr(gqa, "GQAHeadRepeater")
    assert hasattr(gqa, "repeat_kv_pair")


def test_parallax_dispatch_gqa_export() -> None:
    from parallax.dispatch import GQAHeadRepeater, dispatched_repeat_kv, repeat_kv

    assert GQAHeadRepeater is not None
    assert dispatched_repeat_kv is not None
    assert repeat_kv is not None







