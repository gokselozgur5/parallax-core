import numpy as np
import pytest
from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.tensor_ops import dispatched_matmul
from parallax.gradcheck import check_tensor_gradient
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


def test_dispatched_matmul_type_error():
    a = Tensor([1.0, 2.0])
    with pytest.raises(TypeError, match="dispatched_matmul requires Tensor"):
        dispatched_matmul(a, [1.0, 2.0])  # type: ignore[arg-type]


def test_dispatched_matmul_1d():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = dispatched_matmul(a, b)
    c.backward()
    assert np.allclose(c.data, 32.0)
    assert np.allclose(a.grad, [4.0, 5.0, 6.0])
    assert np.allclose(b.grad, [1.0, 2.0, 3.0])
