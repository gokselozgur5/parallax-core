import numpy as np
from parallax.orpo import compute_log_odds


def test_compute_log_odds_monotonic():
    # Higher prob -> higher log odds
    log_probs_high = np.array([-0.1, -0.2])
    log_probs_low = np.array([-2.0, -2.5])

    odds_high = compute_log_odds(log_probs_high)
    odds_low = compute_log_odds(log_probs_low)

    assert odds_high > odds_low
