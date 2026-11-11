"""
Unit tests for BitAttention (BitNet b1.58 Multi-Head Attention Fusion).
"""

import numpy as np
import pytest

from parallax.bit_attention import BitAttention
from parallax.bitnet import BitLinear, quantize_weights_ternary
from parallax.tensor import Tensor, no_grad


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
    with no_grad():
        out_prompt, kv_cache = attn(x_prompt, kv_cache=empty_cache)
    assert kv_cache[0].shape == (B, num_heads, 2, head_dim)
    assert kv_cache[1].shape == (B, num_heads, 2, head_dim)
    np.testing.assert_allclose(out_prompt.data, out_full[:, :2, :], rtol=1e-5, atol=1e-5)

    # Step 1: token 2
    x_tok2 = Tensor(x_full[:, 2:3, :])
    with no_grad():
        out_tok2, kv_cache = attn(x_tok2, kv_cache=kv_cache)
    assert kv_cache[0].shape == (B, num_heads, 3, head_dim)
    assert kv_cache[1].shape == (B, num_heads, 3, head_dim)
    np.testing.assert_allclose(out_tok2.data, out_full[:, 2:3, :], rtol=1e-5, atol=1e-5)

    # Step 2: token 3
    x_tok3 = Tensor(x_full[:, 3:4, :])
    with no_grad():
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


def test_bit_attention_flash_standard_sliding_window_prefill():
    """Verify flash and standard prefill match within 1e-9 with sliding windows."""
    np.random.seed(42)
    B, T, C = 2, 6, 16
    for window in [1, 3, T]:
        std = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=False, use_rope=False)
        fl = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=True, use_rope=False)
        for p_std, p_fl in zip(std.parameters(), fl.parameters()):
            p_fl.data = p_std.data.copy()
        x = np.random.randn(B, T, C)
        out_std = std(Tensor(x.copy())).data
        out_fl = fl(Tensor(x.copy())).data
        np.testing.assert_allclose(out_fl, out_std, atol=1e-9)


def test_bit_attention_flash_standard_sliding_window_cached_decode():
    """Verify flash and standard cached decode match within 1e-9 with sliding windows."""
    np.random.seed(42)
    B, T, C = 1, 4, 16
    for window in [1, 3, T]:
        std = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=False, use_rope=False)
        fl = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=True, use_rope=False)
        for p_std, p_fl in zip(std.parameters(), fl.parameters()):
            p_fl.data = p_std.data.copy()
        x = np.random.randn(B, 1, C)
        head_dim = C // 2
        cache = (np.random.randn(B, 2, 3, head_dim), np.random.randn(B, 2, 3, head_dim))
        with no_grad():
            out_std, _ = std(Tensor(x.copy()), kv_cache=(cache[0].copy(), cache[1].copy()))
            out_fl, _ = fl(Tensor(x.copy()), kv_cache=(cache[0].copy(), cache[1].copy()))
        np.testing.assert_allclose(out_fl.data, out_std.data, atol=1e-9)


def test_bit_attention_cached_grad_enabled_raises():
    """Verify RuntimeError is raised when kv_cache is passed with autograd enabled."""
    attn = BitAttention(embed_dim=16, num_heads=2)
    cache = (np.zeros((1, 2, 0, 8)), np.zeros((1, 2, 0, 8)))
    x = Tensor(np.random.randn(1, 2, 16))
    with pytest.raises(RuntimeError, match="BitAttention with kv_cache is inference-only"):
        attn(x, kv_cache=cache)


def test_bit_attention_gqa_repeat_vs_tile_numerical():
    """Verify GQA repeat grouping against per-head reference; kills tile-vs-repeat mutation."""
    np.random.seed(42)
    B, T, C, num_heads, num_kv_heads = 2, 4, 32, 4, 2
    attn = BitAttention(embed_dim=C, num_heads=num_heads, num_kv_heads=num_kv_heads, use_rope=False)
    x = Tensor(np.random.randn(B, T, C))
    out = attn(x).data

    q = attn.q_proj(x).data.reshape(B, T, num_heads, C // num_heads).transpose(0, 2, 1, 3)
    k = attn.k_proj(x).data.reshape(B, T, num_kv_heads, C // num_heads).transpose(0, 2, 1, 3)
    v = attn.v_proj(x).data.reshape(B, T, num_kv_heads, C // num_heads).transpose(0, 2, 1, 3)
    scale = 1.0 / np.sqrt(C // num_heads)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    ref_heads, tile_heads = [], []
    for h in range(num_heads):
        for heads_list, kv_h in [(ref_heads, h // (num_heads // num_kv_heads)), (tile_heads, h % num_kv_heads)]:
            sc = np.where(mask[None, :, :], -1e9, (q[:, h] @ k[:, kv_h].swapaxes(-1, -2)) * scale)
            w = np.exp(sc - np.max(sc, axis=-1, keepdims=True))
            heads_list.append((w / np.sum(w, axis=-1, keepdims=True)) @ v[:, kv_h])

    ref_out = attn.out_proj(Tensor(np.transpose(np.stack(ref_heads, axis=1), (0, 2, 1, 3)).reshape(B, T, C))).data
    tile_out = attn.out_proj(Tensor(np.transpose(np.stack(tile_heads, axis=1), (0, 2, 1, 3)).reshape(B, T, C))).data
    np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)
    assert np.max(np.abs(out - tile_out)) > 1.0


def test_bit_attention_dq_inverse_rope_effect(monkeypatch):
    """Verify dQ inverse RoPE rotation; kills dQ without inverse RoPE mutation."""
    np.random.seed(42)
    B, T, C, num_heads = 2, 4, 16, 2
    attn = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True, rope_theta=100.0)
    x = Tensor(np.random.randn(B, T, C), requires_grad=True)
    o = attn(x); (o * o).sum().backward()
    true_q_grad = attn.q_proj.weight.grad.copy()

    import parallax.bit_attention as ba
    orig_rope, calls = ba.apply_rope, [0]
    def faulty_rope(x, seq_dim=1, offset=0, conjugate=False, theta=10000.0, interleaved=True):
        if conjugate and calls[0] == 0:
            calls[0] += 1
            return x  # fault: dQ without inverse RoPE
        return orig_rope(x, seq_dim=seq_dim, offset=offset, conjugate=conjugate, theta=theta, interleaved=interleaved)
    monkeypatch.setattr(ba, "apply_rope", faulty_rope)

    attn_f = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True, rope_theta=100.0)
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        getattr(attn_f, p).weight.data = getattr(attn, p).weight.data.copy()
    o_f = attn_f(Tensor(x.data.copy(), requires_grad=True)); (o_f * o_f).sum().backward()
    assert np.max(np.abs(true_q_grad - attn_f.q_proj.weight.grad)) > 0.5


def test_bit_attention_dq_dk_dv_central_difference_reference():
    """Verify dQ, dK, and dV gradients match central difference reference through biases."""
    from parallax.nn import Module

    class Identity(Module):
        def forward(self, x):
            return x

    np.random.seed(42)
    B, T, C, num_heads = 2, 4, 16, 2
    attn = BitAttention(embed_dim=C, num_heads=num_heads, use_rope=True, qkv_bias=True, rope_theta=100.0)
    attn.out_proj = Identity()
    x = Tensor(np.random.randn(B, T, C), requires_grad=False)

    def eval_loss():
        return float(np.sum(attn(x).data ** 2))

    (attn(x) * attn(x)).sum().backward()

    eps = 1e-6
    for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
        fd = np.zeros_like(proj.bias.data)
        for i in range(len(proj.bias.data)):
            orig = proj.bias.data[i]
            proj.bias.data[i] = orig + eps
            lp = eval_loss()
            proj.bias.data[i] = orig - eps
            lm = eval_loss()
            proj.bias.data[i] = orig
            fd[i] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(proj.bias.grad, fd, rtol=1e-4, atol=1e-4)




def test_bit_attention_flash_prefill_empty_cache_then_decode():
    """Verify flash prefill through empty cache followed by cached decode matches standard path (closes #437)."""
    np.random.seed(42)
    B, T, C = 1, 8, 16
    window = 3
    std = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=False, use_rope=False)
    fl = BitAttention(C, 2, num_kv_heads=2, sliding_window=window, use_flash_attn=True, use_rope=False)
    for p_std, p_fl in zip(std.parameters(), fl.parameters()):
        p_fl.data = p_std.data.copy()

    x_prefill = np.random.randn(B, T, C)
    head_dim = C // 2
    empty_cache = (np.zeros((B, 2, 0, head_dim)), np.zeros((B, 2, 0, head_dim)))

    with no_grad():
        out_std, cache_std = std(Tensor(x_prefill), kv_cache=empty_cache)
        out_fl, cache_fl = fl(Tensor(x_prefill), kv_cache=empty_cache)

    np.testing.assert_allclose(out_fl.data, out_std.data, atol=1e-9)

    x_decode = np.random.randn(B, 1, C)
    with no_grad():
        dec_std, _ = std(Tensor(x_decode), kv_cache=cache_std)
        dec_fl, _ = fl(Tensor(x_decode), kv_cache=cache_fl)

    np.testing.assert_allclose(dec_fl.data, dec_std.data, atol=1e-9)
