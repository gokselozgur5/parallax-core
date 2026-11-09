import numpy as np
import pytest
from parallax.attention_sink import RollingKVCache
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.generator import generate_stream


def test_rolling_kv_cache_eviction():
    cache = RollingKVCache(sink_tokens=2, window_size=4)
    # Shape: [B, H, T, D] = [1, 2, 8, 4]
    k = np.arange(8)[None, None, :, None].repeat(2, axis=1).repeat(4, axis=3)
    v = k * 10
    k_evicted, v_evicted = cache.evict(k, v)

    # Sink tokens (first 2): indices 0, 1
    # Window tokens (last 4): indices 4, 5, 6, 7
    expected_indices = [0, 1, 4, 5, 6, 7]
    assert k_evicted.shape == (1, 2, 6, 4)
    assert np.array_equal(k_evicted[0, 0, :, 0], expected_indices)
    assert np.array_equal(v_evicted[0, 0, :, 0], [x * 10 for x in expected_indices])


def test_rolling_kv_cache_below_capacity():
    cache = RollingKVCache(sink_tokens=2, window_size=4)
    k = np.ones((1, 2, 5, 4))
    v = np.ones((1, 2, 5, 4))
    k_out, v_out = cache.evict(k, v)
    assert k_out.shape == (1, 2, 5, 4)
    assert np.array_equal(k_out, k)


def test_generate_stream_with_attention_sink():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    sink = RollingKVCache(sink_tokens=2, window_size=4)

    prompt = [1, 2, 3]
    gen = list(generate_stream(
        model,
        prompt,
        max_new_tokens=15,
        temperature=0.0,
        rolling_cache=sink,
    ))

    assert len(gen) == 15
    for token in gen:
        assert isinstance(token, int)
