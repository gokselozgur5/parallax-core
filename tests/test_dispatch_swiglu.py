"""Unit tests for SwiGLU activation layer and dispatch helpers."""

import numpy as np
import pytest
from parallax.dispatch.swiglu import swiglu_dispatch


@pytest.fixture(autouse=True)
def setup_backend():
    from parallax.backends import registry
    from parallax.backends.numpy_backend import NumpyBackend

    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()



def test_swiglu_dispatch_parity() -> None:
    a = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = swiglu_dispatch(a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    expected = a * sig * b
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_swiglu_dispatch_validation() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        swiglu_dispatch(a, b)
    with pytest.raises(TypeError):
        swiglu_dispatch(np.array([1 + 2j]), np.array([2 + 3j]))


def test_swiglu_backward_dispatch() -> None:
    from parallax.dispatch.swiglu import swiglu_backward_dispatch

    a = np.array([1.0, -1.0], dtype=np.float32)
    b = np.array([2.0, 3.0], dtype=np.float32)
    grad = np.array([1.0, 1.0], dtype=np.float32)
    ga, gb = swiglu_backward_dispatch(grad, a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    silu = a * sig
    dsilu = sig * (1.0 + a * (1.0 - sig))
    np.testing.assert_allclose(gb, grad * silu, atol=1e-5)
    np.testing.assert_allclose(ga, grad * b * dsilu, atol=1e-5)


def test_dispatched_swiglu_tensor_autograd() -> None:
    from parallax.tensor import Tensor
    from parallax.dispatch.swiglu import dispatched_swiglu

    g_data = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float32)
    u_data = np.array([[2.0, 3.0], [-1.0, 0.5]], dtype=np.float32)
    g = Tensor(g_data, requires_grad=True)
    u = Tensor(u_data, requires_grad=True)
    out = dispatched_swiglu(g, u)
    assert out.shape == (2, 2)
    out.sum().backward()
    assert g.grad is not None
    assert u.grad is not None


def test_swiglu_activation_module() -> None:
    from parallax.dispatch.swiglu import SwiGLUActivation
    from parallax.tensor import Tensor

    act = SwiGLUActivation()
    g = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
    u = Tensor(np.array([[0.5, -0.5]], dtype=np.float32))
    out1 = act(g, u)
    assert out1.shape == (1, 2)

    x = Tensor(np.array([[1.0, 2.0, 0.5, -0.5]], dtype=np.float32))
    out2 = act(x)
    assert out2.shape == (1, 2)
    np.testing.assert_allclose(out1.data, out2.data, atol=1e-5)


def test_dispatched_swiglu_ffn_forward() -> None:
    from parallax.dispatch.swiglu import DispatchedSwiGLU
    from parallax.tensor import Tensor

    layer = DispatchedSwiGLU(embed_dim=8, hidden_dim=16)
    x = Tensor(np.random.randn(2, 4, 8).astype(np.float32), requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 4, 8)
    out.sum().backward()
    assert x.grad is not None


def test_dispatched_swiglu_from_swiglu_and_repr() -> None:
    from parallax.models.gpt import SwiGLU
    from parallax.dispatch.swiglu import DispatchedSwiGLU

    orig = SwiGLU(embed_dim=8, hidden_dim=16)
    dispatched = DispatchedSwiGLU.from_swiglu(orig)
    assert dispatched.embed_dim == 8
    assert dispatched.hidden_dim == 16
    assert "embed_dim=8" in dispatched.extra_repr()


def test_swiglu_dispatch_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from parallax.dispatch import swiglu

    a = np.array([0.5, -0.5], dtype=np.float32)
    b = np.array([2.0, -2.0], dtype=np.float32)
    monkeypatch.setattr(swiglu, "c_swiglu", None)
    out = swiglu.swiglu_dispatch(a, b)
    sig = 1.0 / (1.0 + np.exp(-a))
    np.testing.assert_allclose(out, a * sig * b, atol=1e-5)


def test_dispatch_swiglu_package_export() -> None:
    from parallax.dispatch import swiglu as swiglu_mod

    assert hasattr(swiglu_mod, "swiglu_dispatch")
    assert hasattr(swiglu_mod, "swiglu_backward_dispatch")
    assert hasattr(swiglu_mod, "dispatched_swiglu")
    assert hasattr(swiglu_mod, "SwiGLUActivation")
    assert hasattr(swiglu_mod, "DispatchedSwiGLU")


def test_parallax_dispatch_swiglu_export() -> None:
    from parallax.dispatch import DispatchedSwiGLU, SwiGLUActivation, dispatched_swiglu

    assert DispatchedSwiGLU is not None
    assert SwiGLUActivation is not None
    assert dispatched_swiglu is not None









