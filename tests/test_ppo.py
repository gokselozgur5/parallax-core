import numpy as np
from parallax.ppo import compute_gae


def test_compute_gae_single_step_and_td_residual():
    # T = 1 step
    rewards = np.array([1.0], dtype=np.float32)
    values = np.array([0.5, 0.0], dtype=np.float32)  # V(s0)=0.5, V(s1)=0.0
    gamma = 0.99
    gae_lambda = 0.95

    adv, ret = compute_gae(rewards, values, gamma=gamma, gae_lambda=gae_lambda)

    # Expected delta = 1.0 + 0.99 * 0.0 - 0.5 = 0.5
    assert np.isclose(adv[0], 0.5)
    assert np.isclose(ret[0], 1.0)
