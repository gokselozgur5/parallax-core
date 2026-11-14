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


def test_dispatched_matmul_1d_2d_and_2d_1d():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)
    c = dispatched_matmul(a, b)
    c.sum().backward()
    assert np.allclose(a.grad, [12.0, 21.0])
    assert np.allclose(b.grad, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([5.0, 6.0], requires_grad=True)
    z = dispatched_matmul(x, y)
    z.sum().backward()
    assert np.allclose(x.grad, [[5.0, 6.0], [5.0, 6.0]])
    assert np.allclose(y.grad, [4.0, 6.0])


SHAPES = [
    ((4,), (4,)),
    ((3,), (3, 4)),
    ((2, 3), (3,)),
    ((2, 4), (4, 3)),
    ((2, 3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (5, 4, 2)),
]


@pytest.mark.parametrize("s_a,s_b", SHAPES)
def test_dispatched_matmul_matches_tensor_matmul(s_a, s_b):
    np.random.seed(42)
    d_a, d_b = np.random.randn(*s_a), np.random.randn(*s_b)
    t_a1, t_b1 = Tensor(d_a.copy(), requires_grad=True), Tensor(d_b.copy(), requires_grad=True)
    (t_a1 @ t_b1).sum().backward()
    t_a2, t_b2 = Tensor(d_a.copy(), requires_grad=True), Tensor(d_b.copy(), requires_grad=True)
    dispatched_matmul(t_a2, t_b2).sum().backward()
    assert np.allclose(t_a1.grad, t_a2.grad)
    assert np.allclose(t_b1.grad, t_b2.grad)


@pytest.mark.parametrize("s_a,s_b", SHAPES)
def test_dispatched_matmul_finite_differences(s_a, s_b):
    np.random.seed(42)
    t1 = Tensor(np.random.randn(*s_a), requires_grad=True)
    t2 = Tensor(np.random.randn(*s_b), requires_grad=True)
    assert check_tensor_gradient(lambda x, y: dispatched_matmul(x, y).sum(), [t1, t2], eps=1e-5)
