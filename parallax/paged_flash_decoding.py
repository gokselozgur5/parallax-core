"""Generalized FlashDecoding kernel with Paged KV table indexing."""

from typing import List, Tuple
import numpy as np


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
        """Execute split-K FlashDecoding across virtual sequence blocks."""
        k_seq, v_seq = self.gather_paged_blocks(block_table, k_blocks, v_blocks, seq_len)
        dim = q_token.shape[-1]
        scale = 1.0 / np.sqrt(dim)
        chunk_size = max(1, (seq_len + self.num_splits - 1) // self.num_splits)

        partial_m = []
        partial_l = []
        partial_o = []

        for s in range(self.num_splits):
            start = s * chunk_size
            end = min(start + chunk_size, seq_len)
            if start >= seq_len:
                continue
            k_chunk = k_seq[start:end]
            v_chunk = v_seq[start:end]
            scores = np.matmul(q_token, k_chunk.T) * scale
            m_s = np.max(scores)
            exp_s = np.exp(scores - m_s)
            l_s = np.sum(exp_s)
            o_s = np.matmul(exp_s, v_chunk)
            partial_m.append(m_s)
            partial_l.append(l_s)
            partial_o.append(o_s)

        # Global reduction
        global_m = np.max(partial_m)
        global_l = 0.0
        global_o = np.zeros(dim)
        for m_s, l_s, o_s in zip(partial_m, partial_l, partial_o):
            alpha = np.exp(m_s - global_m)
            global_l += l_s * alpha
            global_o += o_s * alpha
        return global_o / max(global_l, 1e-12)
