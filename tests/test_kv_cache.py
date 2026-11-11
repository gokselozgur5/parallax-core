"""
Unit tests for Key-Value (KV) cache autoregressive inference.
"""

import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.generator import generate


def test_kv_cache_matches_non_cached_greedy_logits():
    # Set seed for exact deterministic reproducibility
    np.random.seed(42)
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
        use_rope=True,
    )
    model = ParallaxLM(config)
    prompt = [1, 5, 12, 8]

    # Deterministic greedy generation with cache
    np.random.seed(123)
    out_cached = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=True)

    # Deterministic greedy generation without cache
    np.random.seed(123)
    out_uncached = generate(model, prompt, max_new_tokens=6, temperature=0.0, use_cache=False)

    assert out_cached == out_uncached
    assert len(out_cached) == len(prompt) + 6


def test_kv_cache_step_shape_growth():
    config = ParallaxLMConfig(
        vocab_size=20,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=20,
        use_rope=True,
    )
    model = ParallaxLM(config)

    head_dim = config.embed_dim // config.num_heads
    init_caches = [
        (np.empty((1, config.num_heads, 0, head_dim)),
         np.empty((1, config.num_heads, 0, head_dim)))
        for _ in range(config.num_layers)
    ]

    # Step 1: Prompt length 3
    idx1 = np.array([[2, 4, 6]], dtype=np.int64)
    logits1, caches1 = model(idx1, kv_caches=init_caches)
    assert logits1.shape == (1, 3, config.vocab_size)
    assert caches1[0][0].shape == (1, 2, 3, head_dim)

    # Step 2: Incremental step length 1
    idx2 = np.array([[8]], dtype=np.int64)
    logits2, caches2 = model(idx2, kv_caches=caches1)
    assert logits2.shape == (1, 1, config.vocab_size)
    assert caches2[0][0].shape == (1, 2, 4, head_dim)


def test_generate_eos_token_id_stops_early():
    config = ParallaxLMConfig(
        vocab_size=10,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
    )
    model = ParallaxLM(config)
    prompt = [1, 2]

    greedy_out = generate(model, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
    first_gen = greedy_out[2]

    # Test cached early stopping at first generated token
    out_cached = generate(model, prompt, max_new_tokens=10, temperature=0.0, use_cache=True, eos_token_id=first_gen)
    assert out_cached == [1, 2, first_gen]

    # Test uncached early stopping at first generated token
    out_uncached = generate(model, prompt, max_new_tokens=10, temperature=0.0, use_cache=False, eos_token_id=first_gen)
    assert out_uncached == [1, 2, first_gen]

