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
