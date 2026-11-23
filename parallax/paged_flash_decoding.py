"""Generalized FlashDecoding kernel delegating to flash_decoding_v2."""

from typing import List, Tuple
import numpy as np

from parallax.flash_decoding_v2 import flash_decode_split_k_paged


class PagedFlashDecoder:
    """Combines Split-K FlashDecoding with virtual paged KV block tables."""

    def __init__(self, block_size: int = 16, num_splits: int = 4) -> None:
        self.block_size = int(block_size)
        self.num_splits = int(num_splits)

    def gather_paged_blocks(
        self,
        block_table: List[int],
        k_blocks: np.ndarray,
        v_blocks: np.ndarray,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gather physical page blocks into contiguous sequence K and V tensors."""
        gathered_k = np.concatenate([k_blocks[b] for b in block_table], axis=0)
        gathered_v = np.concatenate([v_blocks[b] for b in block_table], axis=0)
        return gathered_k[:seq_len], gathered_v[:seq_len]

    def decode(
        self,
        q_token: np.ndarray,
        block_table: List[int],
        k_blocks: np.ndarray,
        v_blocks: np.ndarray,
        seq_len: int,
    ) -> np.ndarray:
        """Execute split-K FlashDecoding delegating to flash_decode_split_k_paged."""
        return flash_decode_split_k_paged(
            q_token,
            block_table,
            k_blocks,
            v_blocks,
            seq_len,
            k_splits=self.num_splits,
        )


__all__ = ["PagedFlashDecoder"]
