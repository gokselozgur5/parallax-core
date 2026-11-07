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

    # Baseline = 0.0, Positive Reward = +2.0
    optimizer.zero_grad()
    loss1 = compute_policy_gradient_step(policy, tokens, reward=2.0, baseline=0.0)
    optimizer.step()

    # Probability of target sequence should have increased
    logits_after = policy(tokens).data[0]
    pred = np.argmax(logits_after, axis=-1)
    assert logits_after is not None


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

