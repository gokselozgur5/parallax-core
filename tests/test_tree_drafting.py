from parallax.tree_drafting import TreeDraftNode


def test_tree_draft_node_initialization() -> None:
    node = TreeDraftNode(token_id=42, parent_idx=0)
    assert node.token_id == 42
    assert node.parent_idx == 0
