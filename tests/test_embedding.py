"""Unit tests for DispatchedEmbedding module."""

import numpy as np
import pytest
from parallax.nn import Parameter
from parallax.tensor import Tensor


def test_dispatched_embedding_init():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(num_embeddings=10, embedding_dim=4)
    assert emb.num_embeddings == 10
    assert emb.embedding_dim == 4
    assert emb.weight.shape == (10, 4)
    assert isinstance(emb.weight, Parameter)


def test_dispatched_embedding_forward_shape():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(10, 4)
    idx = Tensor(np.array([[1, 2, 3], [4, 5, 6]]))
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out.data[0, 0], emb.weight.data[1])


def test_dispatched_embedding_padding_idx():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(10, 4, padding_idx=0)
    np.testing.assert_allclose(emb.weight.data[0], np.zeros(4))
    idx = Tensor(np.array([0, 1]))
    out = emb(idx)
    np.testing.assert_allclose(out.data[0], np.zeros(4))


def test_dispatched_embedding_bounds_validation():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(5, 3)
    with pytest.raises(IndexError):
        emb(Tensor(np.array([5])))
    with pytest.raises(IndexError):
        emb(Tensor(np.array([-1])))


def test_dispatched_embedding_backward_grad():
    from parallax.nn.embedding import DispatchedEmbedding
    emb = DispatchedEmbedding(5, 2, padding_idx=0)
    idx = Tensor(np.array([1, 1, 2]))
    out = emb(idx)
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    np.testing.assert_allclose(emb.weight.grad[1], [2.0, 2.0])
    np.testing.assert_allclose(emb.weight.grad[0], [0.0, 0.0])
