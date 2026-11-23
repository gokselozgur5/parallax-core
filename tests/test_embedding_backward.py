"""Bounds validation tests for sparse_embedding_backward."""

import numpy as np
import pytest
from parallax.dispatch.embedding_backward import sparse_embedding_backward


def test_sparse_embedding_backward_rejects_negative_indices():
    grad = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(IndexError, match="out of range"):
        sparse_embedding_backward(grad, np.array([-1, 5]))


def test_sparse_embedding_backward_rejects_out_of_range_bounds():
    grad = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(IndexError, match="out of range"):
        sparse_embedding_backward(grad, np.array([0, 5]), num_embeddings=5)
    with pytest.raises(IndexError, match="padding_idx"):
        sparse_embedding_backward(grad, np.array([0, 1]), padding_idx=-1)
    with pytest.raises(ValueError, match="num_embeddings"):
        sparse_embedding_backward(grad, np.array([0, 1]), num_embeddings=0)


def test_sparse_embedding_backward_valid_bounds():
    idx = np.array([2, 0, 2], dtype=np.int64)
    grad = np.ones((3, 2), dtype=np.float32)
    uniq, acc = sparse_embedding_backward(grad, idx, padding_idx=0, num_embeddings=4)
    np.testing.assert_array_equal(uniq, [0, 2])
    np.testing.assert_allclose(acc, [[0.0, 0.0], [2.0, 2.0]])
