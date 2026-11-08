"""Dynamic Batching and Sequence Length Binning scheduler."""

from typing import List, Tuple
import numpy as np


class SequenceBinner:
    """Bins sequences into buckets to minimize padding overhead in LLM serving."""

    def __init__(self, bucket_boundaries: List[int]) -> None:
        self.bucket_boundaries = sorted(list(set(bucket_boundaries)))

    def find_bucket(self, seq_len: int) -> int:
        """Find the smallest bucket boundary that fits seq_len."""
        for b in self.bucket_boundaries:
            if seq_len <= b:
                return b
        return self.bucket_boundaries[-1]

    def collate_batch(
        self,
        sequences: List[List[int]],
        pad_token_id: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collate variable length sequences into padded array with attention mask."""
        max_len = max(len(s) for s in sequences)
        bucket_len = self.find_bucket(max_len)
        batch_size = len(sequences)
        padded = np.full((batch_size, bucket_len), pad_token_id, dtype=int)
        mask = np.zeros((batch_size, bucket_len), dtype=int)
        for i, seq in enumerate(sequences):
            l = min(len(seq), bucket_len)
            padded[i, :l] = seq[:l]
            mask[i, :l] = 1
        return padded, mask
