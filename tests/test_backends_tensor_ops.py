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
