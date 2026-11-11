from parallax.tree_drafting import TreeDraftNode


def test_tree_draft_node_initialization() -> None:
    node = TreeDraftNode(token_id=42, parent_idx=0)
    assert node.token_id == 42
    assert node.parent_idx == 0

import numpy as np
from parallax.tree_drafting import build_tree_attention_mask


def test_build_tree_attention_mask_ancestors() -> None:
    # Tree: 0 is root, 1 & 2 are children of 0, 3 is child of 1
    parent_indices = [-1, 0, 0, 1]
    mask = build_tree_attention_mask(parent_indices)
    assert mask.shape == (4, 4)
    # Node 3 attends to 3, 1, 0
    assert np.array_equal(mask[3], [1, 1, 0, 1])
    # Node 2 attends to 2, 0
    assert np.array_equal(mask[2], [1, 0, 1, 0])

from parallax.tree_drafting import verify_tree_candidates


def test_verify_tree_candidates_selection() -> None:
    # Tree: 0 (root, token 10) -> 1 (child, token 20), 2 (child, token 30)
    # 1 -> 3 (child, token 40)
    draft_tokens = [10, 20, 30, 40]
    parent_indices = [-1, 0, 0, 1]
    # Target logits: node 0 predicts 20 (accepts 1), node 1 predicts 40 (accepts 3)
    target_logits = np.zeros((4, 100))
    target_logits[0, 20] = 10.0
    target_logits[1, 40] = 10.0
    accepted_path = verify_tree_candidates(
        draft_tokens, parent_indices, target_logits
    )
    assert accepted_path == [10, 20, 40]


def test_tree_drafting_v1_cycles_and_bounds() -> None:
    import pytest
    with pytest.raises(ValueError, match="Cycle or invalid parent"):
        build_tree_attention_mask([0])
    with pytest.raises(ValueError, match="Cycle or invalid parent"):
        build_tree_attention_mask([1, 0])
    with pytest.raises(ValueError, match="Mismatch"):
        verify_tree_candidates([10], [-1, 0], np.zeros((1, 10)))
