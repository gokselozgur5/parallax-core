import numpy as np
import pytest
from parallax.tensor import Tensor


def test_tensor_addition_with_broadcasting():
    # Matrix + Vector: [2, 3] + [3]
    a = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    b = Tensor(np.array([10.0, 20.0, 30.0]))
    c = (a + b).sum()
    c.backward()

    # dL/da should be ones of shape (2, 3)
    assert np.allclose(a.grad, np.ones((2, 3)))
    # dL/db should be summed along axis 0 -> [2, 2, 2]
    assert np.allclose(b.grad, np.array([2.0, 2.0, 2.0]))


def test_tensor_matmul_backward():
    # A @ B -> C
    # A: [2, 3], B: [3, 4]
    np.random.seed(42)
    a_data = np.random.randn(2, 3)
    b_data = np.random.randn(3, 4)

    a = Tensor(a_data)
    b = Tensor(b_data)
    c = a @ b
    loss = c.sum()
    loss.backward()

    # Analytical verification:
    # dL/dA = (dL/dC) @ B^T where dL/dC is ones(2, 4)
    expected_a_grad = np.ones((2, 4)) @ b_data.T
    expected_b_grad = a_data.T @ np.ones((2, 4))

    assert np.allclose(a.grad, expected_a_grad)
    assert np.allclose(b.grad, expected_b_grad)


def test_tensor_batch_matmul_backward():
    # Batched matrix multiplication: [B, T, D] @ [B, D, H]
    np.random.seed(1337)
    x = Tensor(np.random.randn(2, 4, 8))
    w = Tensor(np.random.randn(2, 8, 16))
    out = (x @ w).mean()
    out.backward()

    assert x.grad.shape == (2, 4, 8)
    assert w.grad.shape == (2, 8, 16)


def test_tensor_relu_backward():
    x = Tensor(np.array([[-2.0, 0.5], [3.0, -1.0]]))
    y = x.relu().sum()
    y.backward()

    expected_grad = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(x.grad, expected_grad)
