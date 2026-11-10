"""Speculative Tree-Drafting (Eagle style) candidate verification kernel."""

from typing import List, Optional
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

def verify_tree_candidates(
    draft_tokens: List[int],
    parent_indices: List[int],
    target_logits: np.ndarray,
    context_logits: Optional[np.ndarray] = None,
) -> List[int]:
    """Verify tree candidates greedily against target model predictions and find longest valid path.

    Args:
        draft_tokens: List of proposed draft token IDs corresponding to tree nodes.
        parent_indices: Parent node index for each node, with -1 denoting root nodes.
        target_logits: Logits from target model for each tree candidate node (shape: [n, vocab_size]).
        context_logits: Optional target model logits at the prompt/context boundary prior
            to tree candidates. If provided, roots must match argmax(context_logits) to be accepted.
            If a root is rejected, the target's correction token (argmax) is returned.

    Returns:
        The longest verified token sequence (or [correction_token] if root is rejected).
    """
    n = len(draft_tokens)
    if n == 0:
        return []

    target_preds = np.argmax(target_logits, axis=-1)
    valid_nodes = set()

    context_choice = int(np.argmax(context_logits)) if context_logits is not None else None

    for i in range(n):
        parent = parent_indices[i]
        if parent == -1:
            if context_choice is None or draft_tokens[i] == context_choice:
                valid_nodes.add(i)
        elif parent in valid_nodes and target_preds[parent] == draft_tokens[i]:
            valid_nodes.add(i)

    if not valid_nodes:
        if context_choice is not None:
            return [context_choice]
        return []

    # Find deepest valid branch
    best_path: List[int] = []
    for node in valid_nodes:
        path = []
        curr = node
        while curr != -1 and curr in valid_nodes:
            path.append(draft_tokens[curr])
            curr = parent_indices[curr]
        if len(path) > len(best_path):
            best_path = path[::-1]
    return best_path
