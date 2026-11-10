"""
Unit tests for BitAttention (BitNet b1.58 Multi-Head Attention Fusion).
"""

import numpy as np
import pytest

from parallax.bit_attention import BitAttention
from parallax.bitnet import BitLinear, quantize_weights_ternary
from parallax.tensor import Tensor


def test_bit_attention_projection_types_and_ternary_weights():
    """Verify that Q, K, V, and Out projections are all BitLinear with ternary {-1, 0, 1} quantization."""
    np.random.seed(42)
    embed_dim = 32
    num_heads = 4

    attn = BitAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=True)

    # Check projection types
    assert isinstance(attn.q_proj, BitLinear)
    assert isinstance(attn.k_proj, BitLinear)
    assert isinstance(attn.v_proj, BitLinear)
    assert isinstance(attn.out_proj, BitLinear)

    # Check weights ternary quantization {-1.0, 0.0, 1.0}
    for name, proj in [
        ("q_proj", attn.q_proj),
        ("k_proj", attn.k_proj),
        ("v_proj", attn.v_proj),
        ("out_proj", attn.out_proj),
    ]:
        w = proj.weight.data
        w_ternary, gamma = quantize_weights_ternary(w)
        unique_vals = set(np.unique(w_ternary))
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), f"{name} weights are not strictly ternary"
        assert gamma > 0.0


def test_bit_attention_forward_backward():
    """Test full forward pass and backward pass with Straight-Through Estimator on projections."""
    np.random.seed(1337)
    B, T, C = 2, 4, 16
    num_heads = 2

    attn = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True)
    x = Tensor(np.random.randn(B, T, C), requires_grad=True)

    out = attn(x)
    assert out.shape == (B, T, C)
    assert np.all(np.isfinite(out.data))

    loss = (out * out).sum()
    loss.backward()

    # Verify gradients propagate via STE to projection weights and input tensor
    assert attn.q_proj.weight.grad is not None
    assert attn.q_proj.weight.grad.shape == (C, C)
    assert attn.k_proj.weight.grad is not None
    assert attn.k_proj.weight.grad.shape == (C, C)
    assert attn.v_proj.weight.grad is not None
    assert attn.v_proj.weight.grad.shape == (C, C)
    assert attn.out_proj.weight.grad is not None
    assert attn.out_proj.weight.grad.shape == (C, C)

    assert x.grad is not None
    assert x.grad.shape == (B, T, C)
    assert np.all(np.isfinite(x.grad))


def test_bit_attention_flash_online_softmax_equivalence():
    """Verify that online FlashAttention and standard causal attention yield identical forward outputs."""
    np.random.seed(42)
    B, T, C = 2, 8, 16
    num_heads = 2

    attn_std = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=False, use_flash_attn=False)
    attn_flash = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=False, use_flash_attn=True)

    # Share identical weights
    attn_flash.q_proj.weight.data = attn_std.q_proj.weight.data.copy()
    attn_flash.k_proj.weight.data = attn_std.k_proj.weight.data.copy()
    attn_flash.v_proj.weight.data = attn_std.v_proj.weight.data.copy()
    attn_flash.out_proj.weight.data = attn_std.out_proj.weight.data.copy()

    x_data = np.random.randn(B, T, C)
    x1 = Tensor(x_data.copy())
    x2 = Tensor(x_data.copy())

    out_std = attn_std(x1)
    out_flash = attn_flash(x2)

    np.testing.assert_allclose(out_flash.data, out_std.data, rtol=1e-5, atol=1e-5)


def test_bit_attention_flash_attention_backward():
    """Verify FlashAttention backward pass on BitAttention propagates gradients."""
    np.random.seed(42)
    B, T, C = 2, 8, 16
    num_heads = 2

    attn = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True, use_flash_attn=True)
    x = Tensor(np.random.randn(B, T, C), requires_grad=True)

    out = attn(x)
    loss = (out * out).sum()
    loss.backward()

    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    assert attn.q_proj.weight.grad is not None
    assert np.all(np.isfinite(attn.q_proj.weight.grad))


def test_bit_attention_kv_cache_step_consistency():
    """Verify KV-cache incremental step decoding matches sequence prefill."""
    np.random.seed(42)
    B, T, C = 1, 4, 16
    num_heads = 2
    head_dim = C // num_heads

    attn = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True)

    x_full = np.random.randn(B, T, C)
    out_full = attn(Tensor(x_full)).data

    # Step-by-step decoding with KV-cache
    # Initialize empty cache for prefill
    empty_cache = (
        np.empty((B, num_heads, 0, head_dim), dtype=np.float64),
        np.empty((B, num_heads, 0, head_dim), dtype=np.float64),
    )

    # Step 0: prompt with first 2 tokens
    x_prompt = Tensor(x_full[:, :2, :])
    out_prompt, kv_cache = attn(x_prompt, kv_cache=empty_cache)
    assert kv_cache[0].shape == (B, num_heads, 2, head_dim)
    assert kv_cache[1].shape == (B, num_heads, 2, head_dim)
    np.testing.assert_allclose(out_prompt.data, out_full[:, :2, :], rtol=1e-5, atol=1e-5)

    # Step 1: token 2
    x_tok2 = Tensor(x_full[:, 2:3, :])
    out_tok2, kv_cache = attn(x_tok2, kv_cache=kv_cache)
    assert kv_cache[0].shape == (B, num_heads, 3, head_dim)
    assert kv_cache[1].shape == (B, num_heads, 3, head_dim)
    np.testing.assert_allclose(out_tok2.data, out_full[:, 2:3, :], rtol=1e-5, atol=1e-5)

    # Step 2: token 3
    x_tok3 = Tensor(x_full[:, 3:4, :])
    out_tok3, kv_cache = attn(x_tok3, kv_cache=kv_cache)
    assert kv_cache[0].shape == (B, num_heads, 4, head_dim)
    assert kv_cache[1].shape == (B, num_heads, 4, head_dim)
    np.testing.assert_allclose(out_tok3.data, out_full[:, 3:4, :], rtol=1e-5, atol=1e-5)


def test_bit_attention_causal_non_leakage():
    """Verify future tokens cannot influence past token representations."""
    np.random.seed(999)
    B, T, C = 1, 3, 16
    attn = BitAttention(embed_dim=C, num_heads=2, use_rope=False)

    x1 = np.random.randn(B, T, C)
    x2 = x1.copy()
    x2[0, 2, :] += 50.0  # modify token 2 only

    out1 = attn(Tensor(x1)).data
    out2 = attn(Tensor(x2)).data

    # Tokens 0 and 1 must remain identical
    np.testing.assert_allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-6)
    np.testing.assert_allclose(out1[0, 1, :], out2[0, 1, :], atol=1e-6)
    # Token 2 must differ
    assert not np.allclose(out1[0, 2, :], out2[0, 2, :])


def test_bit_attention_gqa_mode():
    """Verify Grouped-Query Attention (GQA) mode with num_kv_heads < num_heads."""
    np.random.seed(42)
    B, T, C = 2, 4, 32
    num_heads = 4
    num_kv_heads = 2

    attn = BitAttention(
        embed_dim=C,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_rope=True,
    )
    x = Tensor(np.random.randn(B, T, C), requires_grad=True)
    out = attn(x)

    assert out.shape == (B, T, C)
    loss = (out * out).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, C)
    assert attn.k_proj.weight.shape == (num_kv_heads * (C // num_heads), C)
    assert attn.v_proj.weight.shape == (num_kv_heads * (C // num_heads), C)


def test_bit_attention_sliding_window_validation_and_causality():
    """Verify non-positive sliding_window raises ValueError and valid windows preserve causality."""
    with pytest.raises(ValueError):
        BitAttention(embed_dim=16, num_heads=2, sliding_window=0)
    with pytest.raises(ValueError):
        BitAttention(embed_dim=16, num_heads=2, sliding_window=-3)

    np.random.seed(42)
    B, T, C = 1, 6, 16
    for window in [1, 3, T]:
        attn = BitAttention(embed_dim=C, num_heads=2, sliding_window=window, use_rope=False)
        x = np.random.randn(B, T, C)
        x_mod = x.copy()
        x_mod[0, 3:, :] += 10.0
        out = attn(Tensor(x)).data
        out_mod = attn(Tensor(x_mod)).data
        np.testing.assert_allclose(out[0, 0, :], out_mod[0, 0, :], atol=1e-6)
