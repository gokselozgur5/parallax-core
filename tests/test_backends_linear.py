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


def test_dispatched_linear_forward_parity() -> None:
    for bias in [True, False]:
        lin = Linear(4, 3, bias=bias)
        dlin = DispatchedLinear.from_linear(lin)
        x_data = np.random.randn(2, 4)
        x1 = Tensor(x_data.copy())
        x2 = Tensor(x_data.copy())
        out1 = lin(x1)
        out2 = dlin(x2)
        assert out1.shape == (2, 3)
        assert np.allclose(out1.data, out2.data, atol=1e-6)
