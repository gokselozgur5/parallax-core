"""
Unit and gradient verification tests for Mixture-of-Experts (MoE) (M22).
"""

import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.moe import SparseMoE
from parallax.optim import AdamW
from parallax.tensor import Tensor


def test_sparse_moe_forward_and_backward():
    B, T, D = 2, 4, 16
    E, K = 4, 2
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=K, use_swiglu=False)

    x = Tensor(np.random.randn(B, T, D), requires_grad=True)
    out = moe(x)

    assert out.shape == (B, T, D)
    assert moe.last_aux_loss >= 0.0
    assert np.sum(moe.last_expert_counts) == B * T * K

    # Test backward pass
    loss = (out * out).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, D)
    assert moe.gate.weight.grad is not None
    assert moe.gate.weight.grad.shape == (E, D)


def test_sparse_moe_swiglu_experts():
    B, T, D = 1, 3, 16
    E, K = 3, 1
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=K, use_swiglu=True)

    x = Tensor(np.random.randn(B, T, D), requires_grad=True)
    out = moe(x)
    assert out.shape == (B, T, D)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_parallax_lm_with_sparse_moe():
    cfg = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=8,
        use_rope=True,
        num_experts=4,
        top_k_experts=2,
    )
    model = ParallaxLM(cfg)

    # Check that blocks have SparseMoE
    assert isinstance(model.blocks[0].mlp, SparseMoE)
    assert model.blocks[0].mlp.num_experts == 4
    assert model.blocks[0].mlp.top_k == 2

    # Forward pass
    idx = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits = model(idx)
    assert logits.shape == (2, 4, 16)

    # Optimization step
    optimizer = AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = (logits * logits).sum()
    loss.backward()
    optimizer.step()

    # Verify that expert parameters were updated
    e0_weight = model.blocks[0].mlp.experts[0].fc1.weight.data
    assert e0_weight is not None
