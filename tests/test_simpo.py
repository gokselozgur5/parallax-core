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


def _simpo_finite_diff_grads(
    logits_w: np.ndarray,
    targets_w: np.ndarray,
    logits_l: np.ndarray,
    targets_l: np.ndarray,
    beta: float = 2.0,
    gamma_margin: float = 0.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    base_w = logits_w.astype(np.float64, copy=True)
    base_l = logits_l.astype(np.float64, copy=True)

    def token_lp(cand: np.ndarray, targets: np.ndarray) -> np.ndarray:
        shifted = cand - np.max(cand, axis=-1, keepdims=True)
        return shifted[np.arange(len(targets)), targets] - np.log(np.sum(np.exp(shifted), axis=-1))

    def eval_loss() -> float:
        return compute_simpo_loss(
            token_lp(base_w, targets_w), token_lp(base_l, targets_l), beta=beta, gamma_margin=gamma_margin
        )

    num_w = np.zeros_like(base_w)
    for idx in np.ndindex(base_w.shape):
        orig = base_w[idx]
        base_w[idx] = orig + eps
        l_pos = eval_loss()
        base_w[idx] = orig - eps
        l_neg = eval_loss()
        base_w[idx] = orig
        num_w[idx] = (l_pos - l_neg) / (2.0 * eps)

    num_l = np.zeros_like(base_l)
    for idx in np.ndindex(base_l.shape):
        orig = base_l[idx]
        base_l[idx] = orig + eps
        l_pos = eval_loss()
        base_l[idx] = orig - eps
        l_neg = eval_loss()
        base_l[idx] = orig
        num_l[idx] = (l_pos - l_neg) / (2.0 * eps)
    return num_w, num_l


def test_simpo_step_execution_and_finite_gradients():
    np.random.seed(42)
    vocab_size = 15
    logits_w = np.random.randn(4, vocab_size)
    targets_w = np.random.randint(0, vocab_size, size=4)
    logits_l = np.random.randn(3, vocab_size)
    targets_l = np.random.randint(0, vocab_size, size=3)

    loss, grad_w, grad_l = simpo_step(logits_w, targets_w, logits_l, targets_l)
    num_w, num_l = _simpo_finite_diff_grads(logits_w, targets_w, logits_l, targets_l)

    assert isinstance(loss, float)
    assert grad_w.shape == logits_w.shape
    assert grad_l.shape == logits_l.shape
    assert not np.isnan(loss)
    assert not np.isnan(grad_w).any()
    assert not np.isnan(grad_l).any()
    assert np.all(grad_w[np.arange(4), targets_w] < 0.0)
    assert np.all(grad_l[np.arange(3), targets_l] > 0.0)
    np.testing.assert_allclose(grad_w, num_w, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_l, num_l, rtol=1e-5, atol=1e-6)


def test_simpo_step_finite_difference_custom_hyperparams():
    logits_w = np.array([[0.3, -0.4, 0.2, 0.1], [-0.2, 0.5, -0.1, 0.0]], dtype=np.float64)
    targets_w = np.array([1, 2])
    logits_l = np.array([[0.1, 0.2, -0.3, 0.4], [-0.5, 0.0, 0.3, -0.2], [0.2, -0.1, 0.0, 0.1]], dtype=np.float64)
    targets_l = np.array([3, 0, 2])

    _, grad_w, grad_l = simpo_step(logits_w, targets_w, logits_l, targets_l, beta=1.5, gamma_margin=0.8)
    num_w, num_l = _simpo_finite_diff_grads(logits_w, targets_w, logits_l, targets_l, beta=1.5, gamma_margin=0.8)

    np.testing.assert_allclose(grad_w, num_w, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(grad_l, num_l, rtol=1e-6, atol=1e-7)


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

