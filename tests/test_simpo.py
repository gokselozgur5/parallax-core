import numpy as np
from parallax.simpo import compute_simpo_reward


def test_compute_simpo_reward_length_invariant():
    # Sequence of 2 tokens with log prob -1.0 each
    lp_short = np.array([-1.0, -1.0])
    # Sequence of 4 tokens with identical log prob -1.0 each
    lp_long = np.array([-1.0, -1.0, -1.0, -1.0])

    r_short = compute_simpo_reward(lp_short, beta=2.0)
    r_long = compute_simpo_reward(lp_long, beta=2.0)

    # Length normalization means both should have identical reward of 2.0 * (-1.0) = -2.0
    assert np.isclose(r_short, r_long)
    assert np.isclose(r_short, -2.0)
