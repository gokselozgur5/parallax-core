"""Tests for DispatchedEmbedding backward gradient accumulator."""

import numpy as np
import pytest

from python.parallax.dispatch.embedding_backward import embedding_backward


def test_embedding_backward_basic():
    """Verify basic gradient accumulation across indices."""
    num_embeddings = 6
    embedding_dim = 4
    indices = np.array([1, 3, 1, 2], dtype=np.int64)
    grad_output = np.ones((4, embedding_dim), dtype=np.float32)

    grad_weight = embedding_backward(grad_output, indices, num_embeddings, embedding_dim)
    assert grad_weight.shape == (num_embeddings, embedding_dim)
    np.testing.assert_allclose(grad_weight[0], 0.0)
    np.testing.assert_allclose(grad_weight[1], 2.0)
    np.testing.assert_allclose(grad_weight[2], 1.0)
    np.testing.assert_allclose(grad_weight[3], 1.0)


def test_embedding_backward_validation():
    """Verify error checking for invalid dimensions and out of range indices."""
    with pytest.raises(ValueError, match="num_embeddings"):
        embedding_backward(np.zeros((2, 3)), np.array([0, 1]), -1, 3)

    with pytest.raises(ValueError, match="embedding_dim"):
        embedding_backward(np.zeros((2, 3)), np.array([0, 1]), 5, 0)

    with pytest.raises(IndexError, match="out of range"):
        embedding_backward(np.zeros((2, 3)), np.array([0, 10]), 5, 3)

    with pytest.raises(ValueError, match="shape"):
        embedding_backward(np.zeros((2, 4)), np.array([0, 1]), 5, 3)
