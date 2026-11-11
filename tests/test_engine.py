import math
from parallax.engine import Value


def test_sigmoid_overflow_safe():
    # Large negative input should not raise OverflowError and saturate to 0
    v_neg = Value(-1000.0)
    out_neg = v_neg.sigmoid()
    assert out_neg.data == 0.0
    out_neg.backward()
    assert v_neg.grad == 0.0

    # Large positive input should not raise OverflowError and saturate to 1
    v_pos = Value(1000.0)
    out_pos = v_pos.sigmoid()
    assert out_pos.data == 1.0
    out_pos.backward()
    assert v_pos.grad == 0.0

    # Zero input should yield 0.5 with derivative 0.25
    v_zero = Value(0.0)
    out_zero = v_zero.sigmoid()
    assert math.isclose(out_zero.data, 0.5)
    out_zero.backward()
    assert math.isclose(v_zero.grad, 0.25)


def test_exp_overflow_safe():
    # Large positive input clamped safely, no OverflowError
    v_pos = Value(1000.0)
    out_pos = v_pos.exp()
    assert out_pos.data == math.exp(700.0)
    out_pos.backward()
    assert v_pos.grad == out_pos.data

    # Large negative input clamped safely
    v_neg = Value(-1000.0)
    out_neg = v_neg.exp()
    assert out_neg.data == math.exp(-700.0)
    out_neg.backward()
    assert v_neg.grad == out_neg.data

    # Moderate value matches standard math.exp
    v = Value(2.0)
    out = v.exp()
    assert math.isclose(out.data, math.exp(2.0))
    out.backward()
    assert math.isclose(v.grad, math.exp(2.0))
