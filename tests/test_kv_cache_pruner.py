"""
Unit tests for KVCachePruner (StreamingLLM attention sinks + local window + H2O heavy hitters).
"""

import numpy as np
import pytest

from parallax.kv_cache_pruner import KVCachePruner, PruningMetrics


def test_pruner_initialization_and_budget():
    pruner = KVCachePruner(num_sinks=4, window_size=16, heavy_hitters=8)
    assert pruner.budget == 28
    assert pruner.num_sinks == 4
    assert pruner.window_size == 16
    assert pruner.heavy_hitters == 8

    # Test explicit max_budget
    pruner2 = KVCachePruner(num_sinks=4, window_size=16, max_budget=30)
    assert pruner2.budget == 30
    assert pruner2.heavy_hitters == 10

    with pytest.raises(ValueError):
        KVCachePruner(num_sinks=4, window_size=16, max_budget=15)

    with pytest.raises(ValueError):
        KVCachePruner(num_sinks=-1)


def test_prune_no_op_within_budget():
    pruner = KVCachePruner(num_sinks=4, window_size=8, heavy_hitters=4)
    # Budget = 16
    B, H, T, D = 1, 2, 12, 16
    k = np.random.randn(B, H, T, D)
    v = np.random.randn(B, H, T, D)

    k_out, v_out = pruner.prune(k, v)
    assert k_out.shape == (B, H, T, D)
    assert v_out.shape == (B, H, T, D)
    np.testing.assert_array_equal(k, k_out)
    assert pruner.metrics.total_evicted == 0
    assert pruner.metrics.current_cache_size == 12


def test_attention_sinks_and_sliding_window_protection():
    num_sinks = 4
    window_size = 6
    heavy_hitters = 0
    pruner = KVCachePruner(num_sinks=num_sinks, window_size=window_size, heavy_hitters=heavy_hitters)
    # Total budget = 10

    B, H, T, D = 1, 1, 30, 4
    # Set values along seq_dim equal to position index
    k = np.arange(T, dtype=np.float32)[None, None, :, None] * np.ones((B, H, T, D), dtype=np.float32)
    v = k.copy()

    k_pruned, v_pruned = pruner.prune(k, v)
    assert k_pruned.shape[2] == 10
    assert v_pruned.shape[2] == 10

    # Verify sinks (first 4: 0, 1, 2, 3)
    np.testing.assert_array_equal(k_pruned[0, 0, :4, 0], [0, 1, 2, 3])
    # Verify local window (last 6: 24, 25, 26, 27, 28, 29)
    np.testing.assert_array_equal(k_pruned[0, 0, 4:, 0], [24, 25, 26, 27, 28, 29])
    assert pruner.metrics.total_evicted == 20


def test_h2o_heavy_hitter_eviction():
    num_sinks = 2
    window_size = 2
    heavy_hitters = 2
    pruner = KVCachePruner(num_sinks=num_sinks, window_size=window_size, heavy_hitters=heavy_hitters)
    # Budget = 6
    # Sinks: [0, 1]
    # Local window: [8, 9] (for T=10)
    # Candidates in middle: [2, 3, 4, 5, 6, 7]

    T = 10
    B, H, D = 1, 1, 2
    k = np.arange(T, dtype=np.float32)[None, None, :, None] * np.ones((B, H, T, D), dtype=np.float32)
    v = k.copy()

    # Create synthetic attention scores: make token 4 and token 6 have huge attention scores
    attn_scores = np.zeros(T, dtype=np.float32)
    attn_scores[4] = 100.0
    attn_scores[6] = 50.0
    attn_scores[3] = 1.0

    k_pruned, v_pruned = pruner.prune(k, v, attn_scores=attn_scores)

    # Retained tokens must be sinks [0, 1], heavy hitters [4, 6], and window [8, 9]
    expected_indices = [0, 1, 4, 6, 8, 9]
    assert k_pruned.shape[2] == 6
    np.testing.assert_array_equal(k_pruned[0, 0, :, 0], expected_indices)


def test_dynamic_eviction_over_growing_sequence():
    """Simulate autoregressive generation step-by-step to test memory budget bounds and stability."""
    budget = 16
    pruner = KVCachePruner(num_sinks=4, window_size=8, heavy_hitters=4)
    assert pruner.budget == budget

    B, H, D = 1, 2, 8
    # Start with prompt length 10
    k_cache = np.random.randn(B, H, 10, D)
    v_cache = np.random.randn(B, H, 10, D)

    # Simulate 50 autoregressive steps
    for step in range(50):
        # Append 1 new token to KV cache
        new_k = np.random.randn(B, H, 1, D)
        new_v = np.random.randn(B, H, 1, D)
        k_cache = np.concatenate([k_cache, new_k], axis=2)
        v_cache = np.concatenate([v_cache, new_v], axis=2)

        # Generate random attention weights for this step
        curr_len = k_cache.shape[2]
        step_attn = np.random.rand(B, H, 1, curr_len)
        step_attn = step_attn / np.sum(step_attn, axis=-1, keepdims=True)

        pruner.update_attention_scores(layer_idx=0, attn_weights=step_attn)

        # Apply pruning
        k_cache, v_cache = pruner.prune(k_cache, v_cache, layer_idx=0)

        # Assert memory budget is NEVER exceeded
        assert k_cache.shape[2] <= budget
        assert v_cache.shape[2] <= budget
        assert np.all(np.isfinite(k_cache))
        assert np.all(np.isfinite(v_cache))

    # Verify metrics
    assert pruner.metrics.current_cache_size == budget
    assert pruner.metrics.total_evicted > 0
    assert pruner.metrics.eviction_steps > 0


def test_prune_layers_multi_layer():
    pruner = KVCachePruner(num_sinks=2, window_size=2, heavy_hitters=2)
    B, H, T, D = 1, 2, 12, 4
    num_layers = 3

    kv_caches = [
        (np.random.randn(B, H, T, D), np.random.randn(B, H, T, D))
        for _ in range(num_layers)
    ]

    pruned_caches = pruner.prune_layers(kv_caches)
    assert len(pruned_caches) == num_layers
    for pk, pv in pruned_caches:
        assert pk.shape[2] == pruner.budget
        assert pv.shape[2] == pruner.budget
