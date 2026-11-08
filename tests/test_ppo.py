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


from parallax.ppo import compute_ppo_clip_loss


def test_compute_ppo_clip_loss_bounds():
    # Large ratio exceeding 1 + clip_eps
    old_lp = np.array([-1.0, -1.0])
    new_lp = np.array([0.0, 0.0])  # ratio is exp(1.0) ~= 2.718
    advantages = np.array([2.0, 2.0])

    loss, ratios = compute_ppo_clip_loss(new_lp, old_lp, advantages, clip_eps=0.2)

    # When ratio > 1.2 and adv > 0, surr2 = 1.2 * 2.0 = 2.4 clips surr1
    # Loss should be -2.4
    assert np.isclose(loss, -2.4)


from parallax.ppo import ppo_policy_step


def test_ppo_policy_step_shapes_and_finite_gradients():
    np.random.seed(42)
    T = 5
    vocab_size = 20
    logits = np.random.randn(T, vocab_size).astype(np.float32)
    actions = np.random.randint(0, vocab_size, size=T)
    old_lp = np.array([-2.5] * T, dtype=np.float32)
    advantages = np.random.randn(T).astype(np.float32)

    loss, grad = ppo_policy_step(logits, actions, old_lp, advantages, clip_eps=0.2)

    assert isinstance(loss, float)
    assert grad.shape == logits.shape
    assert not np.isnan(loss)
    assert not np.isnan(grad).any()
