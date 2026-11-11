"""
Unit tests for Kahneman-Tversky Optimization (KTO) (M29).
"""

import numpy as np
from parallax.kto import (
    compute_kto_implicit_rewards,
    kto_loss,
    kto_value_function,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.tensor import Tensor


def test_kto_value_function_asymmetry_and_derivatives():
    z = np.array([0.5, -0.5])
    # For desirable, positive z should yield low loss
    loss_d, grad_d = kto_value_function(z, is_desirable=True, lambda_d=1.0)
    assert loss_d[0] < loss_d[1]
    # Gradient of loss w.r.t z must be negative (increasing z reduces loss)
    assert np.all(grad_d < 0.0)

    # For undesirable, positive z should yield high loss
    loss_u, grad_u = kto_value_function(z, is_desirable=False, lambda_u=1.33)
    assert loss_u[0] > loss_u[1]
    # Gradient of loss w.r.t z must be positive (increasing z increases loss)
    assert np.all(grad_u > 0.0)


def test_compute_kto_implicit_rewards_zero_at_init():
    B, T, V = 1, 4, 8
    logits_data = np.random.randn(B, T, V)
    policy_logits = Tensor(logits_data, requires_grad=True)
    targets = np.array([[1, 2, 3, 4]])

    # When policy logits equal reference logits, implicit reward must be strictly 0.0
    r_theta, d_pi = compute_kto_implicit_rewards(
        policy_logits, ref_logits=logits_data, targets=targets, beta=0.1
    )
    np.testing.assert_allclose(r_theta, [0.0], atol=1e-7)
    assert d_pi.shape == (B, T, V)


def test_kto_optimization_increases_desirable_implicit_reward():
    cfg = ParallaxLMConfig(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=6,
    )
    policy = ParallaxLM(cfg)
    optimizer = AdamW(policy.parameters(), lr=0.05)

    desirable_seq = np.array([[1, 2, 3, 4]])
    ref_logits = policy(desirable_seq).data.copy()

    initial_reward = None
    for step in range(4):
        optimizer.zero_grad()
        p_logits = policy(desirable_seq)

        loss, r_val = kto_loss(
            policy_logits=p_logits,
            ref_logits=ref_logits,
            targets=desirable_seq,
            is_desirable=True,
            beta=0.2,
        )
        optimizer.step()

        if step == 0:
            initial_reward = r_val

    # Implicit reward for desirable completion must increase
    assert r_val > initial_reward


