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
