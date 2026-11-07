import pytest
from parallax.engine import Value


def test_value_initialization():
    v = Value(3.5, label="x")
    assert v.data == 3.5
    assert v.grad == 0.0
    assert v.label == "x"
    assert len(v._prev) == 0
    assert v._op == ""


def test_scalar_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0
    assert c._op == "+"
    assert c._prev == {a, b}

    # Python float addition (left and right)
    d = a + 4.0
    assert d.data == 6.0
    e = 4.0 + a
    assert e.data == 6.0


def test_scalar_multiplication():
    a = Value(2.5)
    b = Value(4.0)
    c = a * b
    assert c.data == 10.0
    assert c._op == "*"
    assert c._prev == {a, b}

    # Python float multiplication (left and right)
    d = a * 2.0
    assert d.data == 5.0
    e = 2.0 * a
    assert e.data == 5.0


def test_scalar_subtraction_and_negation():
    a = Value(7.0)
    b = Value(3.0)
    assert (-a).data == -7.0
    assert (a - b).data == 4.0
    assert (b - a).data == -4.0
    assert (a - 2.0).data == 5.0
    assert (10.0 - a).data == 3.0


def test_scalar_division_and_power():
    a = Value(8.0)
    b = Value(2.0)
    assert (a / b).data == 4.0
    assert (b / a).data == 0.25
    assert (a / 2.0).data == 4.0
    assert (16.0 / b).data == 8.0

    assert (b**3).data == 8.0
    assert (a**0.5).data == pytest.approx(2.828427, rel=1e-5)
