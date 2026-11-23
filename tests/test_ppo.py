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


def _ppo_finite_diff_grad(
    logits: np.ndarray,
    actions: np.ndarray,
    old_lp: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float = 0.2,
    eps: float = 1e-6,
) -> np.ndarray:
    base = logits.astype(np.float64, copy=True)
    adv = np.asarray(advantages, dtype=np.float64)
    if len(adv) > 1 and np.std(adv) > 1e-8:
        adv_norm = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)
    else:
        adv_norm = adv

    def eval_loss(cand: np.ndarray) -> float:
        shifted = cand - np.max(cand, axis=-1, keepdims=True)
        lp = shifted[np.arange(len(actions)), actions] - np.log(np.sum(np.exp(shifted), axis=-1))
        return compute_ppo_clip_loss(lp, old_lp, adv_norm, clip_eps=clip_eps)[0]

    num_grad = np.zeros_like(base)
    for idx in np.ndindex(base.shape):
        orig = base[idx]
        base[idx] = orig + eps
        l_pos = eval_loss(base)
        base[idx] = orig - eps
        l_neg = eval_loss(base)
        base[idx] = orig
        num_grad[idx] = (l_pos - l_neg) / (2.0 * eps)
    return num_grad


def test_ppo_policy_step_shapes_and_finite_gradients():
    np.random.seed(42)
    T = 5
    vocab_size = 20
    logits = np.random.randn(T, vocab_size).astype(np.float32)
    actions = np.random.randint(0, vocab_size, size=T)
    old_lp = np.array([-2.5] * T, dtype=np.float32)
    advantages = np.random.randn(T).astype(np.float32)

    loss, grad = ppo_policy_step(logits, actions, old_lp, advantages, clip_eps=0.2)
    num_grad = _ppo_finite_diff_grad(logits, actions, old_lp, advantages, clip_eps=0.2)

    assert isinstance(loss, float)
    assert grad.shape == logits.shape
    assert not np.isnan(loss)
    assert not np.isnan(grad).any()
    assert not np.allclose(grad, 0.0)
    np.testing.assert_allclose(grad, num_grad, rtol=1e-4, atol=1e-5)


def test_ppo_policy_step_finite_difference_gradcheck():
    logits = np.array(
        [[0.4, -0.2, 0.1, 0.0], [-0.3, 0.5, 0.2, -0.1], [2.0, -1.0, -1.0, -1.0]],
        dtype=np.float64,
    )
    actions = np.array([0, 2, 0])
    old_lp = np.array([-1.1, -1.3, -2.0], dtype=np.float64)
    advantages = np.array([2.0, -1.5, 3.0], dtype=np.float64)

    _, grad = ppo_policy_step(logits, actions, old_lp, advantages, clip_eps=0.2)
    num_grad = _ppo_finite_diff_grad(logits, actions, old_lp, advantages, clip_eps=0.2)

    assert grad[0, 0] < 0.0
    assert grad[1, 2] > 0.0
    np.testing.assert_allclose(grad[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(grad, num_grad, rtol=1e-6, atol=1e-7)


def test_compute_gae_keeps_float64_precision():
    rewards = np.array([1.0, 2.0], dtype=np.float64)
    values = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    advantages, returns = compute_gae(rewards, values, gamma=0.9, gae_lambda=0.5)
    assert advantages.dtype == np.float64
    assert returns.dtype == np.float64
    np.testing.assert_allclose(advantages, [2.66, 2.8], rtol=1e-12)
    np.testing.assert_allclose(returns, [3.16, 3.8], rtol=1e-12)

