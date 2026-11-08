import numpy as np
from parallax.orpo import compute_log_odds


def test_compute_log_odds_monotonic():
    # Higher prob -> higher log odds
    log_probs_high = np.array([-0.1, -0.2])
    log_probs_low = np.array([-2.0, -2.5])

    odds_high = compute_log_odds(log_probs_high)
    odds_low = compute_log_odds(log_probs_low)

    assert odds_high > odds_low


from parallax.orpo import compute_odds_ratio_loss


def test_compute_odds_ratio_loss_decreases_as_chosen_dominates():
    # Scenario A: chosen is slightly preferred
    loss_a, sft_a, or_a = compute_odds_ratio_loss(
        log_probs_w=np.array([-0.5, -0.6]),
        log_probs_l=np.array([-1.0, -1.2]),
        lambda_weight=0.5,
    )

    # Scenario B: chosen is overwhelmingly preferred
    loss_b, sft_b, or_b = compute_odds_ratio_loss(
        log_probs_w=np.array([-0.05, -0.05]),
        log_probs_l=np.array([-3.0, -3.5]),
        lambda_weight=0.5,
    )

    assert loss_b < loss_a
    assert or_b < or_a
