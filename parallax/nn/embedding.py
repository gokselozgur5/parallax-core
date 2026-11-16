"""DispatchedEmbedding module routing embedding lookups."""

from __future__ import annotations
from typing import Optional
import numpy as np
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor


class DispatchedEmbedding(Module):
    """Embedding layer supporting backend dispatch and autograd."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        limit = 1.0 / np.sqrt(embedding_dim)
        data = np.random.uniform(-limit, limit, (num_embeddings, embedding_dim))
        self.weight = Parameter(data, label="Embedding.weight")
