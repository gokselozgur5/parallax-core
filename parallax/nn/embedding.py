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
        if padding_idx is not None:
            data[padding_idx] = 0.0
        self.weight = Parameter(data, label="Embedding.weight")

    def forward(self, indices: Tensor | np.ndarray) -> Tensor:
        raw_idx = indices.data if isinstance(indices, Tensor) else indices
        idx_arr = np.asarray(raw_idx, dtype=np.int64)
        out_data = self.weight.data[idx_arr]
        parents = (self.weight,) if isinstance(indices, np.ndarray) else (indices, self.weight)
        out = Tensor(out_data, parents, "Embedding")
        return out
