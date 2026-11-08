import numpy as np
import pytest
from parallax.attention import CausalSelfAttention, apply_rope
from parallax.tensor import Tensor


def test_apply_rope_preserves_norm():
    np.random.seed(42)
    # 2D vector rotation preserves L2 norm
    x = np.random.randn(2, 4, 2, 8)
    rotated = apply_rope(x, seq_dim=1)

    norm_orig = np.linalg.norm(x, axis=-1)
    norm_rotated = np.linalg.norm(rotated, axis=-1)

    assert np.allclose(norm_orig, norm_rotated)


def test_causal_attention_forward_backward():
    np.random.seed(1337)
    B, T, C = 2, 4, 16
    num_heads = 2

    mha = CausalSelfAttention(embed_dim=C, num_heads=num_heads, use_rope=False)
    x = Tensor(np.random.randn(B, T, C))

    out = mha(x)
    assert out.shape == (B, T, C)

    loss = out.sum()
    loss.backward()

    # Verify gradients propagate to projection matrices and input
    assert mha.q_proj.weight.grad is not None
    assert mha.q_proj.weight.grad.shape == (C, C)
    assert mha.out_proj.weight.grad is not None
    assert x.grad.shape == (B, T, C)


def test_causal_mask_non_leakage():
    # Verify that future tokens cannot influence earlier token outputs
    np.random.seed(999)
    B, T, C = 1, 3, 8
    mha = CausalSelfAttention(embed_dim=C, num_heads=2, use_rope=False)

    x1 = np.random.randn(B, T, C)
    x2 = x1.copy()
    # Modify ONLY the last token (t=2) in x2
    x2[0, 2, :] += 50.0

    t1 = Tensor(x1)
    t2 = Tensor(x2)

    out1 = mha(t1)
    out2 = mha(t2)

    # Token 0 and Token 1 outputs MUST be identical between out1 and out2
    assert np.allclose(out1.data[0, 0, :], out2.data[0, 0, :], atol=1e-6)
    assert np.allclose(out1.data[0, 1, :], out2.data[0, 1, :], atol=1e-6)
    # Token 2 can be different
    assert not np.allclose(out1.data[0, 2, :], out2.data[0, 2, :])

def test_apply_rope_conjugate_exact_inverse() -> None:
    np.random.seed(42)
    x = np.random.randn(2, 4, 2, 8)
    rotated = apply_rope(x, seq_dim=1, offset=0, conjugate=False)
    unrotated = apply_rope(rotated, seq_dim=1, offset=0, conjugate=True)
    np.testing.assert_allclose(x, unrotated, rtol=1e-5, atol=1e-5)
