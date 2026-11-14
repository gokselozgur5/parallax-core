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
