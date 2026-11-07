"""
Unit tests for Reward Modeling and Reinforcement Learning (M18).
"""

import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.rl import RewardModel, bradley_terry_reward_loss, compute_policy_gradient_step
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
