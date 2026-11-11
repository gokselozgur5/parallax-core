"""
Unit tests for SwiGLU FFN and Grouped-Query Attention (GQA) (M15).
"""

import numpy as np

from parallax.attention import CausalSelfAttention
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig, SwiGLU, tensor_silu
from parallax.tensor import Tensor


def test_silu_forward_and_backward():
    x = Tensor(np.array([-2.0, 0.0, 2.0]))
    out = tensor_silu(x)

    # SiLU(0) = 0 * 0.5 = 0.0
    assert out.data[1] == 0.0

    # Analytical backward pass check
    out.sum().backward()
    assert x.grad is not None
    # d/dx SiLU(0) = sig(0) + 0 = 0.5
    assert np.isclose(x.grad[1], 0.5, atol=1e-5)


def test_swiglu_forward_and_backward():
    embed_dim = 16
    swiglu = SwiGLU(embed_dim=embed_dim, hidden_dim=32)
    x = Tensor(np.random.randn(2, 4, embed_dim))

    out = swiglu(x)
    assert out.shape == (2, 4, embed_dim)

    loss = out.sum()
    loss.backward()
    assert swiglu.w_gate.weight.grad is not None
    assert swiglu.w_up.weight.grad is not None
    assert swiglu.w_down.weight.grad is not None


def test_gqa_forward_and_backward():
    # 4 query heads, 2 KV heads (group size = 2)
    embed_dim = 16
    num_heads = 4
    num_kv_heads = 2
    gqa = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_rope=True,
    )

    assert gqa.num_queries_per_kv == 2
    assert gqa.k_proj.out_features == 2 * (16 // 4) == 8

    x = Tensor(np.random.randn(2, 6, embed_dim))
    out = gqa(x)
    assert out.shape == (2, 6, embed_dim)

    # Test backward pass accumulation across repeated KV heads
    loss = out.sum()
    loss.backward()
    assert gqa.q_proj.weight.grad is not None
    assert gqa.k_proj.weight.grad is not None
    assert gqa.v_proj.weight.grad is not None


def test_gqa_kv_cache_inference():
    embed_dim = 16
    num_heads = 4
    num_kv_heads = 2
    gqa = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_rope=True,
    )

    head_dim = embed_dim // num_heads
    init_cache = (
        np.empty((1, num_kv_heads, 0, head_dim)),
        np.empty((1, num_kv_heads, 0, head_dim)),
    )

    x_prompt = Tensor(np.random.randn(1, 3, embed_dim))
    out1, cache1 = gqa(x_prompt, kv_cache=init_cache)
    assert cache1 is not None
    # KV cache retains num_kv_heads (2 heads), saving 50% KV memory
    assert cache1[0].shape == (1, 2, 3, 4)

    # Incremental step of length 1
    x_next = Tensor(np.random.randn(1, 1, embed_dim))
    out2, cache2 = gqa(x_next, kv_cache=cache1)
    assert out2.shape == (1, 1, embed_dim)
    assert cache2[0].shape == (1, 2, 4, 4)


def test_parallax_lm_end_to_end_with_swiglu_and_gqa():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=16,
        use_rope=True,
        use_swiglu=True,
        tie_weights=True,
    )
    model = ParallaxLM(cfg)

    idx = np.array([[1, 5, 8, 12]])
    logits = model(idx)
    assert logits.shape == (1, 4, 32)

    loss = logits.sum()
    loss.backward()
    assert model.token_embeddings.weight.grad is not None
