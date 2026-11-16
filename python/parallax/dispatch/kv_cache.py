"""KV Cache tensor buffer manager for pre-allocated autoregressive memory."""

from __future__ import annotations

from typing import Optional
import numpy as np


class KVCacheBuffer:
    """Contiguous pre-allocated buffer for transformer key-value states."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = np.dtype(dtype)
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.k = np.zeros(shape, dtype=self.dtype)
        self.v = np.zeros(shape, dtype=self.dtype)
        self.seq_len = 0

    def memory_usage_bytes(self) -> int:
        """Return total memory allocated for key and value buffers in bytes."""
        return int(self.k.nbytes + self.v.nbytes)
