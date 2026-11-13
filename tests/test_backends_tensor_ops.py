import sys
import types
import numpy as np
import pytest
from parallax.tensor import Tensor
from parallax.backends.tensor_ops import dispatched_matmul


def test_dispatched_matmul_type_error() -> None:
    with pytest.raises(TypeError):
        dispatched_matmul(np.ones((2, 2)), Tensor(np.ones((2, 2))))  # type: ignore[arg-type]


def test_dispatched_matmul_forward_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    mod = types.ModuleType("parallax.backends.dispatch")
    mod.dispatch = lambda op, x, y: (called.append(op) or (x @ y))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parallax.backends.dispatch", mod)
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]))
    out = dispatched_matmul(a, b)
    assert out.shape == (2, 2)
    assert np.allclose(out.data, a.data @ b.data)
    assert called == ["gemm"]


def test_dispatched_matmul_backward_parity() -> None:
    shapes = [((3,), (3,)), ((2, 3), (3, 4)), ((2, 2, 3), (2, 3, 4))]
    for sha, shb in shapes:
        a_data = np.random.randn(*sha).astype(np.float64)
        b_data = np.random.randn(*shb).astype(np.float64)
        t1_a, t1_b = Tensor(a_data.copy(), requires_grad=True), Tensor(b_data.copy(), requires_grad=True)
        t2_a, t2_b = Tensor(a_data.copy(), requires_grad=True), Tensor(b_data.copy(), requires_grad=True)
        (t1_a @ t1_b).sum().backward()
        dispatched_matmul(t2_a, t2_b).sum().backward()
        assert np.allclose(t1_a.grad, t2_a.grad, atol=1e-6)
        assert np.allclose(t1_b.grad, t2_b.grad, atol=1e-6)


def test_dispatched_matmul_finite_diff() -> None:
    a_data = np.random.randn(3, 4).astype(np.float64)
    b_data = np.random.randn(4, 5).astype(np.float64)
    ta, tb = Tensor(a_data.copy(), requires_grad=True), Tensor(b_data.copy(), requires_grad=True)
    dispatched_matmul(ta, tb).sum().backward()
    eps = 1e-5
    for i in range(3):
        for j in range(4):
            ap, am = a_data.copy(), a_data.copy()
            ap[i, j] += eps
            am[i, j] -= eps
            num = float(((ap @ b_data).sum() - (am @ b_data).sum()) / (2 * eps))
            assert abs(float(ta.grad[i, j]) - num) < 1e-5
