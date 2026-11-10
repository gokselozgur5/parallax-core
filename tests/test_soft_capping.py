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
    logits = np.array([2.0, -3.0, 0.5, -1.2, 5.0, -8.0], dtype=np.float64)
    grad_out = np.array([1.0, 0.5, -0.7, 2.0, -1.5, 0.8], dtype=np.float64)
    analytic_grad = capper.backward(grad_out, logits)

    # Numerical finite-difference gradient check
    eps = 1e-6
    num_grad = np.zeros_like(logits)
    for i in range(len(logits)):
        logits_pos = logits.copy()
        logits_pos[i] += eps
        out_pos = capper.forward(logits_pos)

        logits_neg = logits.copy()
        logits_neg[i] -= eps
        out_neg = capper.forward(logits_neg)

        # Loss is sum(out * grad_out)
        loss_pos = np.sum(out_pos * grad_out)
        loss_neg = np.sum(out_neg * grad_out)
        num_grad[i] = (loss_pos - loss_neg) / (2 * eps)

    np.testing.assert_allclose(analytic_grad, num_grad, atol=1e-5, rtol=1e-4)
