"""
Unit tests for alignment and RL suite fixes (Batch 6).
Covers issues:
- #265: exact IPO loss (h_w - h_l - 1/(2 tau))^2 and gradient weight
- #271: pooling last real token in RewardModel with lengths / attention mask
- #272: log-softmax in SimPO and PPO
- #273: float64 log_probs and advantage normalization guard in PPO
- #314: ValueError when kl_coef is non-zero but ref_logits is None
- #315: shape validation in bradley_terry_reward_loss
"""

import numpy as np
import pytest

from parallax.ipo import compute_ipo_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.ppo import compute_ppo_clip_loss, ppo_policy_step
from parallax.rl import RewardModel, bradley_terry_reward_loss, compute_policy_gradient_step
from parallax.simpo import compute_simpo_loss, simpo_step
from parallax.tensor import Tensor


def test_ipo_loss_formula_and_gradient_paper():
    # Issue #265: Verify loss = (h_w - h_l - 1/(2*tau))^2
    tau = 0.25
    h_w = 3.0
    h_l = 0.5
    # diff = (3.0 - 0.5) - 2.0 = 0.5
    # loss = 0.5^2 = 0.25
    # grad_weight = 2.0 * 0.5 / 1 = 1.0
    pi_w = np.array([h_w])
    pi_l = np.array([h_l])
    ref_w = np.array([0.0])
    ref_l = np.array([0.0])

    loss, grad_weight = compute_ipo_loss(pi_w, pi_l, ref_w, ref_l, tau=tau)
    assert np.isclose(loss, 0.25)
    assert np.isclose(grad_weight[0], 1.0)


def test_reward_model_pooling_with_lengths_and_attention_mask():
    # Issue #271: verify hidden state at lengths[b] - 1 is pooled
    cfg = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        use_rope=False,
    )
    rm = RewardModel(cfg)

    # Sequence with padding at the end
    # Batch 0: length 3 (padded to 6)
    # Batch 1: length 5 (padded to 6)
    idx = np.array([
        [1, 2, 3, 0, 0, 0],
        [4, 5, 6, 7, 8, 0],
    ])
    lengths = np.array([3, 5])
    attention_mask = np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 0],
    ])

    out_lengths = rm(idx, lengths=lengths)
    out_mask = rm(idx, attention_mask=attention_mask)
    assert out_lengths.shape == (2, 1)
    assert out_mask.shape == (2, 1)
    assert np.allclose(out_lengths.data, out_mask.data)

    # Test backward pass through lengths pooling
    out_lengths.sum().backward()
    assert rm.reward_head.weight.grad is not None
    assert rm.backbone.token_embeddings.weight.grad is not None


def _log_probs_at_targets(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted[np.arange(len(targets)), targets] - np.log(np.sum(np.exp(shifted), axis=-1))


def test_simpo_log_softmax_numerical_stability():
    # Issue #272: Verify simpo_step handles extreme negative logits without underflow / nan
    vocab_size = 10
    logits_w = np.full((2, vocab_size), -1000.0)
    logits_w[:, 0] = 0.0  # token 0 has prob 1.0, others extremely small
    targets_w = np.array([0, 1])

    logits_l = np.full((2, vocab_size), 0.0)
    targets_l = np.array([0, 1])

    loss, grad_w, grad_l = simpo_step(logits_w, targets_w, logits_l, targets_l)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    assert not np.isnan(grad_w).any()
    assert not np.isnan(grad_l).any()
    assert np.isclose(
        loss,
        compute_simpo_loss(_log_probs_at_targets(logits_w, targets_w), _log_probs_at_targets(logits_l, targets_l)),
    )
    assert grad_w[1, 1] < 0.0 < grad_w[1, 0]
    assert grad_l[0, 0] > 0.0 and grad_l[1, 1] > 0.0


def test_ppo_log_softmax_float64_and_advantage_normalization():
    # Issue #272 & #273: Verify float64 log_probs and adv norm safeguards
    np.random.seed(0)
    T = 4
    vocab_size = 8
    logits = np.random.randn(T, vocab_size)
    actions = np.array([0, 1, 2, 3])
    old_lp = np.array([-1.5, -2.0, -1.8, -2.2])

    def ppo_fd(cand_logits: np.ndarray, act: np.ndarray, olp: np.ndarray, adv_in: np.ndarray) -> np.ndarray:
        adv_f = np.asarray(adv_in, dtype=np.float64)
        adv_n = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8) if len(adv_f) > 1 and adv_f.std() > 1e-8 else adv_f
        eps = 1e-6
        base = cand_logits.astype(np.float64, copy=True)
        out = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            orig = base[idx]
            base[idx] = orig + eps
            lp_pos = _log_probs_at_targets(base, act)
            l_pos = compute_ppo_clip_loss(lp_pos, olp, adv_n)[0]
            base[idx] = orig - eps
            lp_neg = _log_probs_at_targets(base, act)
            l_neg = compute_ppo_clip_loss(lp_neg, olp, adv_n)[0]
            base[idx] = orig
            out[idx] = (l_pos - l_neg) / (2.0 * eps)
        return out

    # Case 1: normal advantages with variance
    adv = np.array([1.0, 2.0, 3.0, 4.0])
    loss, grad = ppo_policy_step(logits, actions, old_lp, adv)
    assert not np.isnan(loss)
    assert not np.isnan(grad).any()
    assert not np.allclose(grad, 0.0)
    np.testing.assert_allclose(grad, ppo_fd(logits, actions, old_lp, adv), rtol=1e-5, atol=1e-6)

    # Case 2: advantages with zero variance (single element or all identical)
    adv_const = np.array([5.0, 5.0, 5.0, 5.0])
    loss_c, grad_c = ppo_policy_step(logits, actions, old_lp, adv_const)
    assert not np.isnan(loss_c)
    assert not np.isnan(grad_c).any()
    assert not np.allclose(grad_c, 0.0)
    np.testing.assert_allclose(grad_c, ppo_fd(logits, actions, old_lp, adv_const), rtol=1e-5, atol=1e-6)

    # Case 3: single advantage
    loss_s, grad_s = ppo_policy_step(logits[:1], actions[:1], old_lp[:1], np.array([2.0]))
    assert not np.isnan(loss_s)
    assert not np.isnan(grad_s).any()
    assert not np.allclose(grad_s, 0.0)
    np.testing.assert_allclose(grad_s, ppo_fd(logits[:1], actions[:1], old_lp[:1], np.array([2.0])), rtol=1e-5, atol=1e-6)


def test_policy_gradient_kl_without_ref_logits_raises():
    # Issue #314: raise ValueError if kl_coef is non-zero but ref_logits is None
    cfg = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        use_rope=False,
    )
    policy = ParallaxLM(cfg)
    tokens = np.array([[1, 2, 3, 4]])

    with pytest.raises(ValueError, match="kl_coef is non-zero but ref_logits was not provided"):
        compute_policy_gradient_step(policy, tokens, reward=1.0, kl_coef=0.1, ref_logits=None)


def test_bradley_terry_shape_mismatch_raises():
    # Issue #315: validate shapes in bradley_terry_reward_loss
    r_chosen = Tensor(np.array([[1.0], [2.0]]))
    r_rejected = Tensor(np.array([[1.5]]))

    with pytest.raises(ValueError, match="Shape mismatch in bradley_terry_reward_loss"):
        bradley_terry_reward_loss(r_chosen, r_rejected)
