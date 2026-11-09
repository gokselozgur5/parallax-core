import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.prune import (
    calculate_model_sparsity,
    prune_2_to_4_structured,
    prune_model,
    prune_unstructured_magnitude,
)


def test_prune_unstructured_magnitude():
    np.random.seed(42)
    weights = np.random.randn(10, 10)
    pruned, mask = prune_unstructured_magnitude(weights, sparsity=0.6)

    zero_count = np.sum(pruned == 0.0)
    assert 58 <= zero_count <= 62
    # Check non-zeros equal original values
    np.testing.assert_allclose(pruned[mask], weights[mask])
    # Check zeros
    assert np.all(pruned[~mask] == 0.0)


def test_prune_2_to_4_structured():
    np.random.seed(42)
    # Shape divisible by 4
    weights = np.random.randn(8, 16)
    pruned, mask = prune_2_to_4_structured(weights)

    # Total sparsity must be strictly 50%
    assert np.sum(pruned == 0.0) == weights.size // 2

    # Check each contiguous 4-element block has exactly 2 zeroes
    blocks = pruned.reshape(-1, 4)
    for b in range(len(blocks)):
        assert np.sum(blocks[b] == 0.0) == 2
        assert np.sum(blocks[b] != 0.0) == 2

    # Check non-zeros match original values
    np.testing.assert_allclose(pruned[mask], weights[mask])


def test_prune_model_end_to_end():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=32,
        intermediate_size=32,
        use_swiglu=True,
    )
    model = ParallaxLM(cfg)

    # Initial model has 0 sparsity
    initial_stats = calculate_model_sparsity(model)
    assert initial_stats["sparsity_ratio"] < 0.01

    # Apply 2:4 structured sparsity to all linear layers
    prune_stats = prune_model(model, method="2:4")

    assert prune_stats["pruned_layers_count"] > 0
    assert prune_stats["sparsity_ratio"] > 0.35  # Linear layers pruned, embedding intact

    # Verify model forward pass remains valid
    idx = np.array([[1, 5, 10]], dtype=np.int64)
    logits = model(idx)
    assert logits.shape == (1, 3, 32)
    assert not np.isnan(logits.data).any()
