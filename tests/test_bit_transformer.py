"""
Unit tests for BitTransformer: Pure 1.58-bit Ternary Transformer Language Model.
"""

import numpy as np
import pytest

from parallax.bitnet import BitLinear, quantize_weights_ternary
from parallax.loss import cross_entropy_loss
from parallax.models.bit_transformer import (
    BitSwiGLU,
    BitTransformerBlock,
    BitTransformerConfig,
    BitTransformerLM,
)
from parallax.tensor import Tensor, no_grad


def test_bit_swiglu_ternary_weights_and_backward():
    """Verify that BitSwiGLU MLP uses BitLinear projections and backpropagates via STE."""
    np.random.seed(42)
    B, T, C = 2, 4, 32
    hidden_dim = 64

    mlp = BitSwiGLU(embed_dim=C, hidden_dim=hidden_dim)

    # Check projection types
    assert isinstance(mlp.w_gate, BitLinear)
    assert isinstance(mlp.w_up, BitLinear)
    assert isinstance(mlp.w_down, BitLinear)

    # Verify ternary quantization {-1, 0, 1}
    for name, proj in [("w_gate", mlp.w_gate), ("w_up", mlp.w_up), ("w_down", mlp.w_down)]:
        w_ternary, gamma = quantize_weights_ternary(proj.weight.data)
        unique_vals = set(np.unique(w_ternary))
        assert unique_vals.issubset({-1.0, 0.0, 1.0}), f"{name} weights are not strictly ternary"
        assert gamma > 0.0

    x = Tensor(np.random.randn(B, T, C), requires_grad=True)
    out = mlp(x)

    assert out.shape == (B, T, C)
    assert np.all(np.isfinite(out.data))

    loss = (out * out).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, T, C)
    assert np.all(np.isfinite(x.grad))

    assert mlp.w_gate.weight.grad is not None
    assert mlp.w_up.weight.grad is not None
    assert mlp.w_down.weight.grad is not None
    assert np.any(mlp.w_gate.weight.grad != 0.0)


def test_bit_transformer_block_forward_backward():
    """Verify BitTransformerBlock forward pass and STE backward gradient flow."""
    np.random.seed(1337)
    B, T, C = 2, 4, 32
    num_heads = 4

    block = BitTransformerBlock(embed_dim=C, num_heads=num_heads, intermediate_size=64)
    x = Tensor(np.random.randn(B, T, C), requires_grad=True)

    out = block(x)
    assert out.shape == (B, T, C)
    assert np.all(np.isfinite(out.data))

    loss = (out * out).sum()
    loss.backward()

    # Input gradient
    assert x.grad is not None
    assert x.grad.shape == (B, T, C)
    assert np.all(np.isfinite(x.grad))

    # BitAttention projection gradients
    assert block.attn.q_proj.weight.grad is not None
    assert block.attn.k_proj.weight.grad is not None
    assert block.attn.v_proj.weight.grad is not None
    assert block.attn.out_proj.weight.grad is not None

    # BitSwiGLU projection gradients
    assert block.mlp.w_gate.weight.grad is not None
    assert block.mlp.w_up.weight.grad is not None
    assert block.mlp.w_down.weight.grad is not None

    # Norm gradients
    assert block.attn_norm.weight.grad is not None
    assert block.mlp_norm.weight.grad is not None


def test_bit_transformer_lm_ternary_parameter_ratio():
    """Verify that >90-95% of model weights are 1.58-bit ternary."""
    config = BitTransformerConfig(
        vocab_size=256,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        intermediate_size=256,
    )
    model = BitTransformerLM(config)

    # Check LM head is BitLinear
    assert isinstance(model.lm_head, BitLinear)

    ratio = model.ternary_ratio()
    # Deep transformer with BitAttention and BitSwiGLU has vast majority of weights ternary
    assert ratio >= 0.90, f"Expected ternary weight ratio >= 0.90, got {ratio:.4f}"


def test_bit_transformer_lm_forward_loss_and_backward():
    """Verify end-to-end forward pass, cross entropy loss computation, and backpropagation."""
    np.random.seed(42)
    vocab_size = 64
    seq_len = 8
    batch_size = 2

    config = BitTransformerConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
        intermediate_size=64,
    )
    model = BitTransformerLM(config)

    # Input tokens and target next tokens
    tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Forward
    logits = model(tokens)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert np.all(np.isfinite(logits.data))

    # Cross Entropy Loss
    loss = cross_entropy_loss(logits, targets)
    assert loss.data > 0.0
    assert np.isfinite(loss.data)

    # Backward
    loss.backward()

    # Embedding grad
    assert model.token_embeddings.weight.grad is not None
    assert np.any(model.token_embeddings.weight.grad != 0.0)

    # LM head grad
    assert model.lm_head.weight.grad is not None
    assert np.any(model.lm_head.weight.grad != 0.0)

    # First layer block grads
    b0 = model.blocks[0]
    assert b0.attn.q_proj.weight.grad is not None
    assert b0.mlp.w_gate.weight.grad is not None


def test_bit_transformer_kv_cache_step_consistency():
    """Verify KV-cached incremental decoding matches full sequence evaluation."""
    np.random.seed(999)
    vocab_size = 32
    seq_len = 6

    config = BitTransformerConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
        intermediate_size=64,
    )
    model = BitTransformerLM(config)

    tokens = np.random.randint(0, vocab_size, size=(1, seq_len))

    # Full sequence forward pass
    logits_full = model(tokens).data

    # Step-by-step decoding with KV-cache
    B = 1
    num_heads = config.num_heads
    head_dim = config.embed_dim // num_heads

    kv_caches = [
        (
            np.empty((B, num_heads, 0, head_dim), dtype=np.float64),
            np.empty((B, num_heads, 0, head_dim), dtype=np.float64),
        )
        for _ in range(config.num_layers)
    ]

    # Prefill first 2 tokens
    prompt = tokens[:, :2]
    with no_grad():
        logits_prompt, kv_caches = model(prompt, kv_caches=kv_caches)
    np.testing.assert_allclose(logits_prompt.data, logits_full[:, :2, :], atol=1e-5, rtol=1e-5)

    # Autoregressively step token by token
    for t in range(2, seq_len):
        tok = tokens[:, t : t + 1]
        with no_grad():
            logits_step, kv_caches = model(tok, kv_caches=kv_caches)
        np.testing.assert_allclose(
            logits_step.data, logits_full[:, t : t + 1, :], atol=1e-5, rtol=1e-5
        )


def test_bit_transformer_autoregressive_generation():
    """Verify generate and generate_stream produce valid token sequences."""
    np.random.seed(42)
    vocab_size = 50
    config = BitTransformerConfig(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
        intermediate_size=64,
    )
    model = BitTransformerLM(config)

    prompt = [1, 5, 12]
    max_new_tokens = 5

    # 1. generate() with KV-cache
    gen_cached = model.generate(
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=5,
        use_cache=True,
    )
    assert len(gen_cached) == len(prompt) + max_new_tokens
    assert gen_cached[: len(prompt)] == prompt
    assert all(0 <= tok < vocab_size for tok in gen_cached)

    # 2. generate() without KV-cache
    gen_uncached = model.generate(
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # Greedy
        use_cache=False,
    )
    assert len(gen_uncached) == len(prompt) + max_new_tokens
    assert gen_uncached[: len(prompt)] == prompt

    # 3. Early stopping with eos_token_id
    eos_token_id = 9
    # Force eos token to be sampled by overriding generator or testing branch
    tokens_stream = list(
        model.generate_stream(
            prompt_tokens=prompt,
            max_new_tokens=10,
            temperature=1.0,
            eos_token_id=eos_token_id,
        )
    )
    assert len(tokens_stream) <= 10


def test_bit_transformer_generate_max_new_tokens_edge_cases():
    """Verify max_new_tokens=0,1,5 cached/uncached match length and negative raises ValueError."""
    config = BitTransformerConfig(vocab_size=32, max_seq_len=16, embed_dim=16, num_heads=2, num_layers=1, intermediate_size=32)
    model = BitTransformerLM(config)
    prompt = [1, 2, 3]

    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=-1, use_cache=True)
    with pytest.raises(ValueError):
        model.generate(prompt, max_new_tokens=-1, use_cache=False)

    for n in [0, 1, 5]:
        c = model.generate(prompt, max_new_tokens=n, use_cache=True)
        u = model.generate(prompt, max_new_tokens=n, use_cache=False)
        assert len(c) == len(prompt) + n
        assert len(u) == len(prompt) + n


def test_bit_transformer_block_mlp_residual_presence():
    """Verify MLP residual connection is present; kills residual-removed fault."""
    np.random.seed(42)
    B, T, C = 2, 4, 16
    block = BitTransformerBlock(embed_dim=C, num_heads=2, intermediate_size=32)
    x = Tensor(np.random.randn(B, T, C))

    x1 = x + block.attn(block.attn_norm(x))
    expected_with_residual = (x1 + block.mlp(block.mlp_norm(x1))).data
    expected_without_residual = block.mlp(block.mlp_norm(x1)).data

    actual = block(x).data
    np.testing.assert_allclose(actual, expected_with_residual, atol=1e-5)
    assert np.linalg.norm(actual - expected_without_residual) > 1.0


def test_bit_transformer_block_mlp_residual_zero_weights():
    """Verify residual passes input when MLP projections are zeroed."""
    np.random.seed(42)
    B, T, C = 1, 3, 16
    block = BitTransformerBlock(embed_dim=C, num_heads=2, intermediate_size=32)
    block.mlp.w_down.weight.data.fill(0.0)
    x = Tensor(np.random.randn(B, T, C))
    actual = block(x).data
    expected = (x + block.attn(block.attn_norm(x))).data
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert np.linalg.norm(actual) > 0.5  # would be 0 if residual was dropped


