"""Speculative Tree-Drafting (Eagle style) candidate verification kernel."""

from typing import List, Tuple
import numpy as np


class TreeDraftNode:
    """Represents a candidate token branch node in the speculative tree."""

    def __init__(self, token_id: int, parent_idx: int = -1) -> None:
        self.token_id = int(token_id)
        self.parent_idx = int(parent_idx)

def build_tree_attention_mask(parent_indices: List[int]) -> np.ndarray:
    """Construct causal tree attention mask where each node attends to its ancestors."""
    n = len(parent_indices)
    mask = np.zeros((n, n), dtype=int)
    for i in range(n):
        curr = i
        while curr != -1:
            mask[i, curr] = 1
            curr = parent_indices[curr]
    return mask
