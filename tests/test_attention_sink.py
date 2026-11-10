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
    assert sink.total_seen == 17


def test_rolling_kv_cache_total_seen_tracking():
    cache = RollingKVCache(sink_tokens=2, window_size=4)
    assert cache.total_seen == 0

    k = np.zeros((1, 1, 8, 4))
    v = np.zeros((1, 1, 8, 4))
    k, v = cache.evict(k, v)
    assert cache.total_seen == 8
    assert k.shape[2] == 6

    # Adding 1 incremental token
    k = np.zeros((1, 1, 7, 4))
    v = np.zeros((1, 1, 7, 4))
    k, v = cache.evict(k, v)
    assert cache.total_seen == 9

    cache.reset()
    assert cache.total_seen == 0


def test_retained_window_distinct_and_increasing_rope_positions():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    sink = RollingKVCache(sink_tokens=2, window_size=4)

    repeated_token = 5
    idx = np.array([[repeated_token] * 6], dtype=np.int64)
    head_dim = config.embed_dim // config.num_heads
    kv_caches = [
        (np.empty((1, config.num_heads, 0, head_dim), dtype=np.float64),
         np.empty((1, config.num_heads, 0, head_dim), dtype=np.float64))
    ]

    _, kv_caches = model(idx, kv_caches=kv_caches)
    kv_caches = sink.evict_layers(kv_caches)

    for step in range(8):
        idx_next = np.array([[repeated_token]], dtype=np.int64)
        _, kv_caches = model(idx_next, kv_caches=kv_caches, rope_offset=sink.total_seen)
        kv_caches = sink.evict_layers(kv_caches)

        k_window = kv_caches[0][0][0, 0, 2:, :]
        # 1. Verify all window keys are distinct despite repeated token
        for i in range(4):
            for j in range(i + 1, 4):
                assert not np.allclose(k_window[i], k_window[j])

        # 2. Verify strictly increasing rotary positions (step = +1.0 rad)
        angles = np.arctan2(k_window[:, 1], k_window[:, 0])
        diffs = np.diff(np.unwrap(angles))
        assert np.allclose(diffs, 1.0)


def test_generate_stream_rope_offsets_do_not_collapse(monkeypatch):
    import parallax.attention as attn_module

    recorded_offsets = []
    orig_apply_rope = attn_module.apply_rope

    def record_rope(x, seq_dim=1, offset=0, **kwargs):
        recorded_offsets.append(offset)
        return orig_apply_rope(x, seq_dim=seq_dim, offset=offset, **kwargs)

    monkeypatch.setattr(attn_module, "apply_rope", record_rope)

    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    sink = RollingKVCache(sink_tokens=2, window_size=6)

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    _ = list(generate_stream(
        model,
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        rolling_cache=sink,
    ))

    # Every step invokes apply_rope for Q and K once per layer
    step_offsets = recorded_offsets[::2]
    expected_offsets = [0, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert step_offsets == expected_offsets
