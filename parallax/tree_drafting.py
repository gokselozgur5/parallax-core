"""Speculative Tree-Drafting module delegating to tree_drafting_v2."""

from typing import List, Optional
import numpy as np

from parallax.tree_drafting_v2 import (
    build_vectorized_tree_attention_mask,
    verify_tree_candidates_v2,
)


class TreeDraftNode:
    """Represents a candidate token branch node in the speculative tree."""

    def __init__(self, token_id: int, parent_idx: int = -1) -> None:
        self.token_id = int(token_id)
        self.parent_idx = int(parent_idx)


def build_tree_attention_mask(parent_indices: List[int]) -> np.ndarray:
    """Construct causal tree attention mask delegating to tree_drafting_v2."""
    try:
        mask, _ = build_vectorized_tree_attention_mask(parent_indices)
        return mask
    except ValueError as e:
        if "Cycle detected" in str(e):
            raise ValueError(f"Cycle or invalid parent detected: {e}") from e
        raise


def verify_tree_candidates(
    draft_tokens: List[int],
    parent_indices: List[int],
    target_logits: np.ndarray,
    context_logits: Optional[np.ndarray] = None,
) -> List[int]:
    """Verify tree candidates delegating to verify_tree_candidates_v2."""
    try:
        res = verify_tree_candidates_v2(
            draft_tokens,
            parent_indices,
            target_logits,
            context_logits=context_logits,
        )
        if isinstance(res, tuple):
            return res[0]
        return res
    except ValueError as e:
        if "Length mismatch" in str(e):
            raise ValueError(f"Mismatch between draft tokens and parent indices: {e}") from e
        raise


__all__ = [
    "TreeDraftNode",
    "build_tree_attention_mask",
    "verify_tree_candidates",
]
