"""Unit tests for MoE, MoD, and ECR routing suite fixes (Batch 6).

Covers:
- Issue #278: SparseMoE router receives non-zero gradient when top_k=1.
- Issue #279: SparseMoE auxiliary load-balancing loss exposed as Tensor / backpropagated.
- Issue #284: MoDRouter causal inference routing without future token leakage.
- Issue #285: ExpertChoiceRouter clamps capacity when capacity factor exceeds token count.
"""

import numpy as np
import pytest

from parallax.ecr import ExpertChoiceRouter
from parallax.gradcheck import check_tensor_gradient
from parallax.mod import MoDRouter, mod_block_forward
from parallax.moe import SparseMoE
from parallax.tensor import Tensor


def test_sparse_moe_top1_router_gradient_nonzero():
    """Verify router receives non-zero gradient when top_k=1 (#278)."""
    np.random.seed(42)
    B, T, D, E = 2, 3, 8, 4
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=1, aux_loss_coef=0.0)

    x = Tensor(np.random.randn(B, T, D), requires_grad=True)
    out = moe(x)

    loss = (out * out).sum()
    loss.backward()

    assert moe.router.weight.grad is not None
    assert moe.gate.weight.grad is not None
    assert np.any(moe.router.weight.grad != 0.0)
    assert np.all(np.isfinite(moe.router.weight.grad))


def test_sparse_moe_top1_gradcheck():
    """Verify finite-differences analytical gradient match for top_k=1 (#278)."""
    np.random.seed(42)
    B, T, D, E = 2, 2, 4, 3
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=1, aux_loss_coef=0.0)
    x = Tensor(np.random.randn(B, T, D), requires_grad=True)

    passed = check_tensor_gradient(
        lambda inp, gw: (moe(inp) * moe(inp)).sum(),
        [x, moe.gate.weight],
    )
    assert passed is True


def test_sparse_moe_aux_loss_tensor_wiring():
    """Verify aux_loss is exposed as a Tensor and propagates gradient into router (#279)."""
    np.random.seed(42)
    B, T, D, E, K = 2, 4, 8, 4, 2
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=K, aux_loss_coef=0.05)

    x = Tensor(np.random.randn(B, T, D), requires_grad=True)
    out = moe(x)

    assert isinstance(moe.aux_loss, Tensor)
    assert moe.aux_loss.data > 0.0
    assert np.isclose(float(moe.aux_loss.data), moe.last_aux_loss)

    # Backprop through objective that wires aux_loss
    total_loss = out.sum() + moe.aux_loss
    total_loss.backward()

    assert moe.router.weight.grad is not None
    assert np.any(moe.router.weight.grad != 0.0)
    assert np.all(np.isfinite(moe.router.weight.grad))


def test_sparse_moe_aux_loss_gradcheck():
    """Verify aux_loss tensor gradient matches finite differences (#279)."""
    np.random.seed(42)
    B, T, D, E, K = 2, 2, 4, 3, 2
    moe = SparseMoE(embed_dim=D, num_experts=E, top_k=K, aux_loss_coef=0.02)
    x = Tensor(np.random.randn(B, T, D), requires_grad=True)

    def aux_objective(inp, gw):
        moe.gate.weight = gw
        moe.router = gw
        _ = moe(inp)
        return moe.aux_loss

    passed = check_tensor_gradient(
        aux_objective,
        [x, moe.gate.weight],
    )
    assert passed is True


def test_mod_causal_routing_no_future_leakage():
    """Verify causal routing decisions at position t do not depend on future tokens > t (#284)."""
    np.random.seed(42)
    hidden_dim = 8
    seq_len = 5
    router = MoDRouter(hidden_dim=hidden_dim, capacity_factor=0.5, threshold=0.5)

    x = np.random.randn(seq_len, hidden_dim).astype(np.float32)

    # Route prefix of length 3 causally
    x_prefix = x[:3]
    prefix_indices, prefix_weights, prefix_scores = router.route_tokens(x_prefix, causal=True)

    # Route full sequence causally
    full_indices, full_weights, full_scores = router.route_tokens(x, causal=True)

    # Scores and decisions for the first 3 tokens must be identical
    np.testing.assert_allclose(prefix_scores, full_scores[:3], rtol=1e-5, atol=1e-5)
    prefix_selected_in_full = [idx for idx in full_indices if idx < 3]
    assert list(prefix_indices) == prefix_selected_in_full


def test_mod_block_forward_causal():
    """Verify mod_block_forward supports causal mode and identity bypass (#284)."""
    seq_len, hidden_dim = 6, 4
    x = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    router = MoDRouter(hidden_dim=hidden_dim, capacity_factor=0.5, threshold=0.4)

    def dummy_block(h):
        return h * 2.0

    output, selected_indices = mod_block_forward(x, router, dummy_block, causal=True)

    assert output.shape == x.shape
    unselected = [i for i in range(seq_len) if i not in selected_indices]
    for idx in unselected:
        assert np.allclose(output[idx], x[idx])


def test_ecr_capacity_clamped_to_token_count():
    """Verify compute_expert_capacity and route_tokens clamp capacity when exceeding tokens (#285)."""
    num_tokens = 3
    num_experts = 2
    capacity_factor = 4.0  # (3 * 4) / 2 = 6 > num_tokens (3)
    router = ExpertChoiceRouter(num_experts=num_experts, capacity_factor=capacity_factor)

    cap = router.compute_expert_capacity(num_tokens)
    assert cap == num_tokens
    assert cap <= num_tokens

    # Verify route_tokens runs without shape mismatch or index error
    gating_scores = np.random.uniform(0.1, 1.0, (num_tokens, num_experts))
    indices, weights = router.route_tokens(gating_scores)

    assert indices.shape == (num_experts, num_tokens)
    assert weights.shape == (num_experts, num_tokens)

    # Combine outputs
    dim = 4
    expert_outs = np.ones((num_experts, num_tokens, dim))
    combined = router.combine_expert_outputs(
        expert_outs, indices, weights, num_tokens=num_tokens, dim=dim
    )
    assert combined.shape == (num_tokens, dim)
    assert np.all(np.isfinite(combined))
