from parallax.soft_capping import LogitSoftCapper


def test_logit_soft_capper_init() -> None:
    capper = LogitSoftCapper(soft_cap=50.0)
    assert capper.soft_cap == 50.0

import numpy as np


def test_soft_capper_forward_bounds() -> None:
    capper = LogitSoftCapper(soft_cap=30.0)
    extreme_logits = np.array([-1000.0, 0.0, 1000.0])
    capped = capper.forward(extreme_logits)
    assert np.isclose(capped[0], -30.0)
    assert np.isclose(capped[1], 0.0)
    assert np.isclose(capped[2], 30.0)

def test_soft_capper_backward_gradient_check() -> None:
    capper = LogitSoftCapper(soft_cap=10.0)
    logits = np.array([2.0, -3.0])
    grad_out = np.ones_like(logits)
    grad = capper.backward(grad_out, logits)
    # At logits=0, tanh=0 -> grad=1. For non-zero, grad < 1
    assert np.all(grad < 1.0)
    assert np.all(grad > 0.0)
