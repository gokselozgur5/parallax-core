import numpy as np
import pytest
from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
from parallax.core.matmul import dispatched_matmul
from parallax.tensor import Tensor


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()


def test_dispatched_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    out = dispatched_matmul(a, b)
    assert np.allclose(out.data, a.data @ b.data)
