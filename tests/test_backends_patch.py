import pytest
from parallax.backends import registry
from parallax.backends.linear import DispatchedLinear
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.patch import disable_native, enable_native
from parallax.nn import Linear, Module


class MLP(Module):
    def __init__(self) -> None:
        self.fc1 = Linear(4, 8)
        self.layers = [Linear(8, 8), Linear(8, 2)]


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()


def test_enable_and_disable_single() -> None:
    lin = Linear(4, 2)
    native = enable_native(lin)
    assert isinstance(native, DispatchedLinear)
    restored = disable_native(native)
    assert type(restored) is Linear


def test_enable_and_disable_container() -> None:
    model = MLP()
    enable_native(model)
    assert isinstance(model.fc1, DispatchedLinear)
    assert all(isinstance(ly, DispatchedLinear) for ly in model.layers)
    disable_native(model)
    assert type(model.fc1) is Linear
    assert all(type(ly) is Linear for ly in model.layers)


import numpy as np
from parallax.tensor import Tensor


def test_param_identity_and_parity() -> None:
    lin = Linear(4, 3, bias=True)
    orig_w, orig_b = lin.weight, lin.bias
    x = Tensor(np.random.randn(2, 4))
    y_orig = lin(x)
    native = enable_native(lin)
    assert native.weight is orig_w
    assert native.bias is orig_b
    assert np.allclose(native(x).data, y_orig.data, atol=1e-6)
    restored = disable_native(native)
    assert restored.weight is orig_w
    assert restored.bias is orig_b
    assert np.allclose(restored(x).data, y_orig.data, atol=1e-6)
