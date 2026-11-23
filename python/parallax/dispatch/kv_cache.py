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

    def reset(self) -> None:
        """Reset sequence position and zero out buffer contents."""
        self.seq_len = 0
        self.k.fill(0)
        self.v.fill(0)

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
        if k_val.shape != v_val.shape:
            raise ValueError(f"k_val shape {k_val.shape} must match v_val shape {v_val.shape}")
        pos = self.seq_len if start_pos is None else start_pos
        if pos < 0 or pos > self.seq_len:
            raise ValueError(f"start_pos {pos} must be 0 <= start_pos <= seq_len {self.seq_len}")
        num_toks = k_val.shape[2]
        if pos + num_toks > self.max_seq_len:
            raise ValueError(
                f"Update length {pos + num_toks} exceeds maximum sequence length {self.max_seq_len}"
            )
        self.k[:, :, pos : pos + num_toks, :] = k_val
        self.v[:, :, pos : pos + num_toks, :] = v_val
        self.seq_len = max(self.seq_len, pos + num_toks)
        return self.get_view(self.seq_len)


class KVCacheManager:
    """Manages multi-layer key-value cache tensor buffers."""

    def __init__(
        self, num_layers: int, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, dtype: np.dtype | type = np.float32
    ) -> None:
        self.num_layers = num_layers
        self.layers = [
            KVCacheBuffer(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype)
            for _ in range(num_layers)
        ]

    def update(
        self, layer_idx: int, k_val: np.ndarray, v_val: np.ndarray, start_pos: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.layers[layer_idx].update(k_val, v_val, start_pos=start_pos)

    def memory_usage_bytes(self) -> int:
        return sum(layer.memory_usage_bytes() for layer in self.layers)

    def reset(self) -> None:
        """Reset all managed layer buffers."""
        for layer in self.layers:
            layer.reset()


def create_kv_cache(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    num_layers: int = 1,
    dtype: np.dtype | type = np.float32,
) -> KVCacheBuffer | KVCacheManager:
    """Create a single-layer buffer or multi-layer cache manager."""
    if num_layers > 1:
        return KVCacheManager(
            num_layers, batch_size, max_seq_len, num_heads, head_dim, dtype=dtype
        )
    return KVCacheBuffer(batch_size, max_seq_len, num_heads, head_dim, dtype=dtype)


__all__ = [
    "KVCacheBuffer",
    "KVCacheManager",
    "create_kv_cache",
]
