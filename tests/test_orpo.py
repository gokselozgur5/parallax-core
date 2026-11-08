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


from parallax.orpo import orpo_step


def test_orpo_step_execution_and_gradient_shapes():
    vocab_size = 20
    logits_w = np.random.randn(5, vocab_size)
    targets_w = np.random.randint(0, vocab_size, size=5)
    logits_l = np.random.randn(4, vocab_size)
    targets_l = np.random.randint(0, vocab_size, size=4)

    loss, grad_w, grad_l = orpo_step(logits_w, targets_w, logits_l, targets_l)

    assert isinstance(loss, float)
    assert grad_w.shape == logits_w.shape
    assert grad_l.shape == logits_l.shape
    assert not np.isnan(loss)
