import pytest
from parallax.engine import Value


def test_backward_simple_add_mul():
    # f(x, y) = (x + y) * x
    # df/dx = 2x + y
    # df/dy = x
    x = Value(3.0, label="x")
    y = Value(4.0, label="y")
    z = (x + y) * x
    z.backward()

    assert z.data == 21.0
    assert x.grad == pytest.approx(10.0)  # 2*3 + 4 = 10
    assert y.grad == pytest.approx(3.0)   # x = 3


def test_backward_gradient_accumulation_multi_branch():
    # Test gradient accumulation (a + a + a)
    # y = 3 * a
    # dy/da = 3
    a = Value(5.0)
    y = a + a + a
    y.backward()

    assert y.data == 15.0
    assert a.grad == pytest.approx(3.0)


def test_backward_power_and_division():
    # f(a, b) = a^3 / b
    # df/da = 3 * a^2 / b = 3 * 4 / 4 = 3
    # df/db = -a^3 / b^2 = -8 / 16 = -0.5
    a = Value(2.0)
    b = Value(4.0)
    f = (a**3) / b
    f.backward()

    assert f.data == 2.0
    assert a.grad == pytest.approx(3.0)
    assert b.grad == pytest.approx(-0.5)


def test_backward_subtraction_and_negation():
    # f(x, y) = -(x - y) = -x + y
    # df/dx = -1, df/dy = 1
    x = Value(10.0)
    y = Value(6.0)
    f = -(x - y)
    f.backward()

    assert f.data == -4.0
    assert x.grad == pytest.approx(-1.0)
    assert y.grad == pytest.approx(1.0)


def test_backward_deep_graph_iterative_topo():
    x = Value(1.0)
    curr = x
    for _ in range(5000):
        curr = curr + 1.0
    curr.backward()
    assert x.grad == pytest.approx(1.0)
