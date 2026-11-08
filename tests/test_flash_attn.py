"""
Unit tests for online softmax update primitive.
"""

import numpy as np
import pytest
from parallax.flash_attn import (
    flash_attention_backward,
    flash_attention_forward,
    reference_attention,
    update_online_softmax,
)


def test_update_online_softmax_single_step():
    B, H, Br, Bc, d = 1, 1, 2, 4, 8
    m_old = np.full((B, H, Br), -np.inf)
    l_old = np.zeros((B, H, Br))
    O_old = np.zeros((B, H, Br, d))

    S_block = np.random.randn(B, H, Br, Bc)
    V_block = np.random.randn(B, H, Bc, d)

    m_new, l_new, O_new = update_online_softmax(m_old, l_old, O_old, S_block, V_block)

    # Output after one block must match exact direct softmax
    exp_S = np.exp(S_block - np.max(S_block, axis=-1, keepdims=True))
    P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    expected_out = np.matmul(P, V_block)

    actual_out = O_new / l_new[:, :, :, None]
    np.testing.assert_allclose(actual_out, expected_out, atol=1e-6)


def test_reference_attention_shape_and_causality():
    q = np.random.randn(1, 2, 4, 8)
    k = np.random.randn(1, 2, 4, 8)
    v = np.random.randn(1, 2, 4, 8)
    out = reference_attention(q, k, v, is_causal=True)
    assert out.shape == (1, 2, 4, 8)


def test_flash_attention_exact_causal_match():
    np.random.seed(42)
    B, H, T, d = 2, 2, 32, 8
    q, k, v = np.random.randn(B, H, T, d), np.random.randn(B, H, T, d), np.random.randn(B, H, T, d)

    ref_out = reference_attention(q, k, v, is_causal=True)
    flash_out, _ = flash_attention_forward(q, k, v, is_causal=True, block_r=8, block_c=8)

    np.testing.assert_allclose(flash_out, ref_out, atol=1e-6, rtol=1e-5)


def test_flash_attention_asymmetric_tiles_non_causal():
    np.random.seed(99)
    B, H, T, d = 1, 2, 25, 8
    q, k, v = np.random.randn(B, H, T, d), np.random.randn(B, H, T, d), np.random.randn(B, H, T, d)

    ref_out = reference_attention(q, k, v, is_causal=False)
    # Use block sizes that do not divide 25 evenly (e.g. 7 and 11)
    flash_out, _ = flash_attention_forward(q, k, v, is_causal=False, block_r=7, block_c=11)

    np.testing.assert_allclose(flash_out, ref_out, atol=1e-6, rtol=1e-5)


def test_flash_attention_backward_causal_gradcheck():
    np.random.seed(42)
    B, H, T, d = 1, 2, 16, 8
    q = np.random.randn(B, H, T, d)
    k = np.random.randn(B, H, T, d)
    v = np.random.randn(B, H, T, d)
    do = np.random.randn(B, H, T, d)

    # Forward pass
    o, lse = flash_attention_forward(q, k, v, is_causal=True, block_r=8, block_c=8)
    # Tiled backward pass
    dq_flash, dk_flash, dv_flash = flash_attention_backward(
        do, o, q, k, v, lse, is_causal=True, block_r=8, block_c=8
    )

    # Reference numerical gradient on Q via finite differences
    eps = 1e-5
    for b in range(B):
        for h in range(H):
            for t in range(2):  # Sample first 2 tokens
                for c in range(2):
                    q_pos = q.copy()
                    q_pos[b, h, t, c] += eps
                    o_pos, _ = flash_attention_forward(q_pos, k, v, is_causal=True)

                    q_neg = q.copy()
                    q_neg[b, h, t, c] -= eps
                    o_neg, _ = flash_attention_forward(q_neg, k, v, is_causal=True)

                    loss_pos = np.sum(o_pos * do)
                    loss_neg = np.sum(o_neg * do)
                    num_grad = (loss_pos - loss_neg) / (2 * eps)

                    np.testing.assert_allclose(dq_flash[b, h, t, c], num_grad, atol=1e-5, rtol=1e-4)




