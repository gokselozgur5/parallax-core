"""DispatchedEmbedding module routing embedding lookups through backend dispatch."""

from __future__ import annotations
from typing import Optional
import numpy as np
from parallax.backends.tensor_ops import dispatched_gather
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor


class DispatchedEmbedding(Module):
    """Embedding layer routing lookups through dispatched_gather with NumPy fallback."""

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
        return dispatched_gather(self.weight, indices, padding_idx=self.padding_idx)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor | np.ndarray,
        freeze: bool = True,
        padding_idx: Optional[int] = None,
    ) -> DispatchedEmbedding:
        data = embeddings.data if isinstance(embeddings, Tensor) else np.asarray(embeddings)
        num_emb, emb_dim = data.shape
        mod = cls(num_emb, emb_dim, padding_idx=padding_idx)
        mod.weight = Parameter(data.copy(), label="Embedding.weight")
        mod.weight.requires_grad = not freeze
        return mod

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}"

    def __repr__(self) -> str:
        return f"DispatchedEmbedding({self.extra_repr()})"
