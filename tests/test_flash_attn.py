"""
Unit tests for online softmax update primitive.
"""

import numpy as np
import pytest
from parallax.flash_attn import update_online_softmax


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
