"""
Unit tests for Reward Modeling and Reinforcement Learning (M18).
"""

import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.rl import (
    RewardModel,
    bradley_terry_reward_loss,
    compute_policy_gradient_step,
    compute_sequence_log_probs,
    dpo_loss,
)
from parallax.tensor import Tensor


def test_reward_model_forward_and_backward():
    cfg = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        use_rope=True,
    )
    rm = RewardModel(cfg)

    # Batch of 2 sequences of length 4
    idx = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    rewards = rm(idx)

    assert rewards.shape == (2, 1)
    assert isinstance(rewards, Tensor)

    loss = rewards.sum()
    loss.backward()

    # Gradients should flow into both the reward_head and backbone
    assert rm.reward_head.weight.grad is not None
    assert rm.backbone.token_embeddings.weight.grad is not None


def test_bradley_terry_preference_loss_optimization():
    # Simulate reward scores where chosen is initially scored lower than rejected
    r_chosen = Tensor(np.array([[1.0], [0.5]]))
    r_rejected = Tensor(np.array([[2.5], [3.0]]))

    loss = bradley_terry_reward_loss(r_chosen, r_rejected)
    assert loss.data > 0.0

    loss.backward()
    # Gradient on r_chosen should be negative (increase reward for chosen)
    assert np.all(r_chosen.grad < 0.0)
    # Gradient on r_rejected should be positive (decrease reward for rejected)
    assert np.all(r_rejected.grad > 0.0)


def test_policy_gradient_step():
    cfg = ParallaxLMConfig(
        vocab_size=10,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
    )
    policy = ParallaxLM(cfg)
    optimizer = AdamW(policy.parameters(), lr=0.05)

    tokens = np.array([[1, 2, 3, 4]])

    def _seq_log_prob(model: ParallaxLM, toks: np.ndarray) -> float:
        logits = model(toks).data[0]
        targets = toks[0, 1:]
        length = len(targets)
        max_l = np.max(logits[:length], axis=-1, keepdims=True)
        exp_l = np.exp(logits[:length] - max_l)
        log_p = (logits[:length] - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))
        return float(np.sum(log_p[np.arange(length), targets]))

    log_prob_before = _seq_log_prob(policy, tokens)

    # Baseline = 0.0, Positive Reward = +2.0
    optimizer.zero_grad()
    loss1 = compute_policy_gradient_step(policy, tokens, reward=2.0, baseline=0.0)
    optimizer.step()

    # Log-probability of target sequence must strictly increase after positive update
    log_prob_after_pos = _seq_log_prob(policy, tokens)
    assert log_prob_after_pos > log_prob_before

    # Negative update (reward = -2.0) must decrease log-probability
    optimizer.zero_grad()
    loss2 = compute_policy_gradient_step(policy, tokens, reward=-2.0, baseline=0.0)
    optimizer.step()

    log_prob_after_neg = _seq_log_prob(policy, tokens)
    assert log_prob_after_neg < log_prob_after_pos


def test_compute_sequence_log_probs_and_gradients():
    B, T, V = 2, 4, 6
    logits_data = np.random.randn(B, T, V)
    logits = Tensor(logits_data, requires_grad=True)
    targets = np.random.randint(0, V, size=(B, T))

    log_probs, dlogits = compute_sequence_log_probs(logits, targets)
    assert log_probs.shape == (B,)
    assert dlogits.shape == (B, T, V)
    # Check that analytical gradient matches finite-difference check on log probs
    eps = 1e-5
    for b in range(B):
        for t in range(T):
            for v in range(V):
                logits_pos = logits_data.copy()
                logits_pos[b, t, v] += eps
                lp_pos, _ = compute_sequence_log_probs(Tensor(logits_pos), targets)

                logits_neg = logits_data.copy()
                logits_neg[b, t, v] -= eps
                lp_neg, _ = compute_sequence_log_probs(Tensor(logits_neg), targets)

                num_grad = (lp_pos[b] - lp_neg[b]) / (2 * eps)
                np.testing.assert_allclose(dlogits[b, t, v], num_grad, atol=1e-5, rtol=1e-4)


def test_dpo_loss_optimization_and_implicit_rewards():
    cfg = ParallaxLMConfig(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=6,
    )
    policy = ParallaxLM(cfg)
    optimizer = AdamW(policy.parameters(), lr=0.05)

    chosen_ids = np.array([[1, 2, 3, 4]])
    rejected_ids = np.array([[1, 5, 6, 7]])

    # Generate initial reference logits (freeze as reference model)
    ref_chosen_logits = policy(chosen_ids).data.copy()
    ref_rejected_logits = policy(rejected_ids).data.copy()

    # Step DPO multiple times
    initial_margin = None
    for step in range(5):
        optimizer.zero_grad()
        p_chosen = policy(chosen_ids)
        p_rejected = policy(rejected_ids)

        loss, r_w, r_l = dpo_loss(
            p_chosen,
            p_rejected,
            ref_chosen_logits,
            ref_rejected_logits,
            chosen_targets=chosen_ids,
            rejected_targets=rejected_ids,
            beta=0.2,
        )

        optimizer.step()

        margin = r_w - r_l
        if step == 0:
            initial_margin = margin

    # After optimization, the implicit reward margin r_chosen - r_rejected must increase
    assert margin > initial_margin
    assert r_w > r_l


def test_policy_gradient_step_kl_penalty_finite_differences():
    np.random.seed(99)
    cfg = ParallaxLMConfig(
        vocab_size=6,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=6,
    )
    policy = ParallaxLM(cfg)
    tokens = np.array([[1, 2, 3]])
    ref_logits = np.random.randn(1, 3, 6)

    policy.zero_grad()
    loss0 = compute_policy_gradient_step(
        policy, tokens, reward=1.5, baseline=0.5, kl_coef=0.3, ref_logits=ref_logits
    )
    analytic_grad = policy.token_embeddings.weight.grad.copy()

    eps = 1e-5
    for idx_tok in [1, 2]:
        for d in range(3):
            orig = policy.token_embeddings.weight.data[idx_tok, d]
            policy.token_embeddings.weight.data[idx_tok, d] = orig + eps
            lp = compute_policy_gradient_step(
                policy, tokens, reward=1.5, baseline=0.5, kl_coef=0.3, ref_logits=ref_logits
            )
            policy.token_embeddings.weight.data[idx_tok, d] = orig - eps
            lm = compute_policy_gradient_step(
                policy, tokens, reward=1.5, baseline=0.5, kl_coef=0.3, ref_logits=ref_logits
            )
            policy.token_embeddings.weight.data[idx_tok, d] = orig
            num_g = (lp - lm) / (2.0 * eps)
            np.testing.assert_allclose(analytic_grad[idx_tok, d], num_g, atol=1e-5, rtol=1e-4)

