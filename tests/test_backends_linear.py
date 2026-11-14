import pytest
from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
import numpy as np
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear, Parameter
from parallax.tensor import Tensor


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()
