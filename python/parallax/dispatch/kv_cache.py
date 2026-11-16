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

    def get_view(self, seq_len: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return views of active key and value tensors up to seq_len."""
        length = self.seq_len if seq_len is None else seq_len
        return self.k[:, :, :length, :], self.v[:, :, :length, :]

    def update(
        self,
        k_val: np.ndarray,
        v_val: np.ndarray,
        start_pos: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update buffer at start_pos and return active views."""
        pos = self.seq_len if start_pos is None else start_pos
        num_toks = k_val.shape[2]
        if pos + num_toks > self.max_seq_len:
            raise ValueError(
                f"Update length {pos + num_toks} exceeds maximum sequence length {self.max_seq_len}"
            )
        self.k[:, :, pos : pos + num_toks, :] = k_val
        self.v[:, :, pos : pos + num_toks, :] = v_val
        self.seq_len = max(self.seq_len, pos + num_toks)
        return self.get_view(self.seq_len)
