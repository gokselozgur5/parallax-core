"""Tests for DispatchedEmbedding backward gradient accumulator."""

import numpy as np
import pytest

from parallax.dispatch.embedding_backward import (
    DispatchedEmbeddingBackward,
    accumulate_embedding_backward,
    embedding_backward,
    sparse_embedding_backward,
)


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


def test_embedding_backward_padding_idx():
    """Verify that padding_idx rows receive zero gradients."""
    num_embeddings = 5
    embedding_dim = 3
    indices = np.array([0, 1, 2, 1], dtype=np.int64)
    grad_output = np.full((4, embedding_dim), 3.0, dtype=np.float32)

    grad_weight = embedding_backward(
        grad_output, indices, num_embeddings, embedding_dim, padding_idx=1
    )
    np.testing.assert_allclose(grad_weight[0], 3.0)
    np.testing.assert_allclose(grad_weight[1], 0.0)
    np.testing.assert_allclose(grad_weight[2], 3.0)


def test_embedding_backward_out_buffer():
    """Verify in-place accumulation into preallocated out buffer."""
    num_embeddings = 4
    embedding_dim = 2
    indices = np.array([0, 2], dtype=np.int64)
    grad_output = np.ones((2, embedding_dim), dtype=np.float32)
    buf = np.ones((num_embeddings, embedding_dim), dtype=np.float32)

    res = embedding_backward(
        grad_output, indices, num_embeddings, embedding_dim, out=buf
    )
    assert res is buf
    np.testing.assert_allclose(buf[0], 2.0)
    np.testing.assert_allclose(buf[1], 1.0)
    np.testing.assert_allclose(buf[2], 2.0)


def test_sparse_embedding_backward():
    """Verify sparse unique indices and gradient accumulation."""
    indices = np.array([2, 5, 2], dtype=np.int64)
    grad_output = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    uniq_idx, grad_slices = sparse_embedding_backward(grad_output, indices)
    np.testing.assert_array_equal(uniq_idx, [2, 5])
    np.testing.assert_allclose(grad_slices[0], [6.0, 8.0])
    np.testing.assert_allclose(grad_slices[1], [3.0, 4.0])


def test_accumulate_embedding_backward():
    """Verify autograd gradient accumulation into Parameter tensor."""
    from parallax.nn import Parameter
    from parallax.tensor import Tensor

    weight = Parameter(np.zeros((6, 3), dtype=np.float32))
    indices = Tensor(np.array([1, 2, 1], dtype=np.int64))
    grad_output = np.ones((3, 3), dtype=np.float32)

    accumulate_embedding_backward(grad_output, indices, weight, padding_idx=2)
    assert weight.grad is not None
    np.testing.assert_allclose(weight.grad[1], 2.0)
    np.testing.assert_allclose(weight.grad[2], 0.0)


def test_dispatched_embedding_backward_class():
    """Verify DispatchedEmbeddingBackward class wrapper interface."""
    indices = np.array([0, 1, 0], dtype=np.int64)
    grad_output = np.ones((3, 2), dtype=np.float32)
    gw = DispatchedEmbeddingBackward.compute(grad_output, indices, 3, 2)
    assert gw.shape == (3, 2)
    np.testing.assert_allclose(gw[0], [2.0, 2.0])
    np.testing.assert_allclose(gw[1], [1.0, 1.0])
    inst = DispatchedEmbeddingBackward()
    assert "DispatchedEmbeddingBackward" in repr(inst)


def test_dispatch_embedding_backward_all():
    """Verify __all__ contains public functions and class."""
    import sys

    eb = sys.modules["parallax.dispatch.embedding_backward"]
    assert hasattr(eb, "__all__")
    expected = {
        "DispatchedEmbeddingBackward",
        "accumulate_embedding_backward",
        "embedding_backward",
        "sparse_embedding_backward",
    }
    assert expected.issubset(set(eb.__all__))


def test_dispatch_package_exports():
    """Verify symbols are exposed via parallax.dispatch package."""
    import parallax.dispatch as pd

    assert hasattr(pd, "DispatchedEmbeddingBackward")
    assert hasattr(pd, "accumulate_embedding_backward")
    assert hasattr(pd, "embedding_backward")
    assert hasattr(pd, "sparse_embedding_backward")
