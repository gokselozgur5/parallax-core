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


from parallax.simpo import compute_simpo_loss


def test_compute_simpo_loss_margin_satisfaction():
    lp_w = np.array([-0.1, -0.1])
    lp_l = np.array([-2.0, -2.0])

    # Reward w: 2.0 * -0.1 = -0.2
    # Reward l: 2.0 * -2.0 = -4.0
    # Reward diff: 3.8
    # If gamma=0.5, margin satisfied (3.8 > 0.5) -> small loss
    loss_satisfied = compute_simpo_loss(lp_w, lp_l, beta=2.0, gamma_margin=0.5)

    # If gamma=5.0, margin violated (3.8 < 5.0) -> high loss
    loss_violated = compute_simpo_loss(lp_w, lp_l, beta=2.0, gamma_margin=5.0)

    assert loss_satisfied < loss_violated


from parallax.simpo import simpo_step


def test_simpo_step_execution_and_finite_gradients():
    vocab_size = 15
    logits_w = np.random.randn(4, vocab_size)
    targets_w = np.random.randint(0, vocab_size, size=4)
    logits_l = np.random.randn(3, vocab_size)
    targets_l = np.random.randint(0, vocab_size, size=3)

    loss, grad_w, grad_l = simpo_step(logits_w, targets_w, logits_l, targets_l)

    assert isinstance(loss, float)
    assert grad_w.shape == logits_w.shape
    assert grad_l.shape == logits_l.shape
    assert not np.isnan(loss)
    assert not np.isnan(grad_w).any()
    assert not np.isnan(grad_l).any()


def test_simpo_step_empty_targets_raises_value_error():
    import pytest

    vocab_size = 10
    logits_w = np.random.randn(0, vocab_size)
    targets_w = np.array([], dtype=np.int64)
    logits_l = np.random.randn(2, vocab_size)
    targets_l = np.array([1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="targets_w cannot be empty"):
        simpo_step(logits_w, targets_w, logits_l, targets_l)

    logits_w_valid = np.random.randn(2, vocab_size)
    targets_w_valid = np.array([0, 1], dtype=np.int64)
    logits_l_empty = np.random.randn(0, vocab_size)
    targets_l_empty = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="targets_l cannot be empty"):
        simpo_step(logits_w_valid, targets_w_valid, logits_l_empty, targets_l_empty)

