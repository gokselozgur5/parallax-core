"""DispatchedEmbedding module routing embedding lookups."""

from __future__ import annotations
from typing import Optional
import numpy as np
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


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
        if np.any(idx_arr < 0) or np.any(idx_arr >= self.num_embeddings):
            raise IndexError("Index out of range for DispatchedEmbedding")
        out_data = self.weight.data[idx_arr]
        parents = (self.weight,) if isinstance(indices, np.ndarray) else (indices, self.weight)
        out = Tensor(out_data, parents, "Embedding")
        if not out.requires_grad:
            return out

        def _backward() -> None:
            if out.grad is None or not self.weight.requires_grad:
                return
            grad_w = np.zeros_like(self.weight.data)
            np.add.at(grad_w, idx_arr, out.grad)
            if self.padding_idx is not None:
                grad_w[self.padding_idx] = 0.0
            _accumulate_grad(self.weight, grad_w)

        out._backward = _backward
        return out

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
