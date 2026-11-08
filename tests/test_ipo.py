from parallax.ipo import IPOConfig


def test_ipo_config_target_margin() -> None:
    config = IPOConfig(tau=0.2)
    assert config.tau == 0.2
    assert config.target_margin == 2.5

import numpy as np
from parallax.ipo import compute_ipo_loss


def test_ipo_loss_zero_when_margin_satisfied() -> None:
    tau = 0.5
    target_margin = 1.0 / (2.0 * tau)  # 1.0
    # Let tau * (h_w - h_l) == target_margin
    # 0.5 * (2.0 - 0.0) == 1.0
    pi_w = np.array([2.0])
    pi_l = np.array([0.0])
    ref_w = np.array([0.0])
    ref_l = np.array([0.0])
    loss, grad_weight = compute_ipo_loss(pi_w, pi_l, ref_w, ref_l, tau=tau)
    assert np.isclose(loss, 0.0, atol=1e-6)
    assert np.isclose(grad_weight[0], 0.0, atol=1e-6)

from parallax.ipo import ipo_step


def test_ipo_step_gradient_direction() -> None:
    logits = np.zeros((2, 10))
    chosen = np.array([2, 5])
    rejected = np.array([3, 7])
    grad_weight = np.array([0.5, 0.5])
    grad = ipo_step(logits, chosen, rejected, grad_weight)
    # Chosen should have negative gradients (-0.5) to ascend logprob
    assert grad[0, 2] == -0.5
    assert grad[0, 3] == 0.5
    assert grad[1, 5] == -0.5
    assert grad[1, 7] == 0.5
