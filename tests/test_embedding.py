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
