"""Unit tests for Tree Drafting v2: dynamic branching, vectorized attention mask, pruning, and matrix-based candidate verification."""

import numpy as np
import pytest

from parallax.tree_drafting import verify_tree_candidates as verify_v1
from parallax.tree_drafting_v2 import (
    TreeDraftNodeV2,
    TreeDraftTopology,
    TreeSpeculativeGenerator,
    build_vectorized_tree_attention_mask,
    generate_dynamic_eagle_tree,
    prune_tree_topology,
    validate_tree_topology,
    verify_tree_candidates_v2,
)


def test_tree_draft_node_v2_dataclass():
    node = TreeDraftNodeV2(node_id=1, token_id=101, parent_idx=0, depth=1, score=0.95)
    assert node.node_id == 1
    assert node.token_id == 101
    assert node.parent_idx == 0
    assert node.depth == 1
    assert pytest.approx(node.score) == 0.95


def test_build_vectorized_tree_attention_mask():
    # Tree: 0 is root, 1 & 2 are children of 0, 3 is child of 1
    parent_indices = [-1, 0, 0, 1]
    mask, ancestors = build_vectorized_tree_attention_mask(parent_indices)
    assert mask.shape == (4, 4)
    assert ancestors.shape == (4, 4)
    
    # Node 3 attends to 0, 1, 3
    assert np.array_equal(mask[3], [1, 1, 0, 1])
    # Node 2 attends to 0, 2
    assert np.array_equal(mask[2], [1, 0, 1, 0])
    # Node 0 attends to 0
    assert np.array_equal(mask[0], [1, 0, 0, 0])
    # Ancestor matrix check
    assert bool(ancestors[3, 0]) is True
    assert bool(ancestors[3, 1]) is True
    assert bool(ancestors[3, 2]) is False
    assert bool(ancestors[3, 3]) is True


def test_generate_dynamic_eagle_tree_topology():
    # Dynamic tree with root=10, depth=2, top_k_per_node=2
    # Layer 0: 1 node (id 0, token 10)
    # Layer 1: 2 nodes (id 1 token 101, id 2 token 102)
    # Layer 2: 4 nodes (children of 1: id 3, 4; children of 2: id 5, 6)
    # Total nodes = 1 + 2 + 4 = 7
    tree = generate_dynamic_eagle_tree(root_token_id=10, depth=2, top_k_per_node=2)
    assert isinstance(tree, TreeDraftTopology)
    assert len(tree.tokens) == 7
    assert tree.tokens[0] == 10
    assert tree.parent_indices[0] == -1
    assert tree.attention_mask.shape == (7, 7)
    assert tree.ancestor_matrix.shape == (7, 7)
    
    # Check that children of node 1 (ids 3, 4) have parent 1
    assert tree.parent_indices[3] == 1
    assert tree.parent_indices[4] == 1
    # Node 3 attends to root (0), parent (1), and self (3)
    assert bool(tree.ancestor_matrix[3, 0]) is True
    assert bool(tree.ancestor_matrix[3, 1]) is True
    assert bool(tree.ancestor_matrix[3, 3]) is True
    assert bool(tree.ancestor_matrix[3, 2]) is False


def test_prune_tree_topology():
    # Tokens: [10, 20, 21, 22, 30]
    # Parent: [-1,  0,  0,  0,  1] (root 0 has 3 children: 1, 2, 3)
    tokens = [10, 20, 21, 22, 30]
    parents = [-1, 0, 0, 0, 1]
    scores = [1.0, 0.8, 0.4, 0.2, 0.7]

    # top_k_per_node = 2 should keep children 1 (0.8) and 2 (0.4), pruning child 3 (0.2)
    pruned_toks, new_parents, kept = prune_tree_topology(
        tokens, parents, scores, max_nodes=10, top_k_per_node=2
    )
    # Child 3 (index 3) pruned, but index 4 (child of 1) should be retained
    assert 3 not in kept
    assert 0 in kept and 1 in kept and 4 in kept
    assert len(pruned_toks) == 4
    # Root has new index 0, child 1 has new index 1
    assert new_parents[0] == -1


def test_matrix_verification_equivalence_with_v1():
    # Compare verify_tree_candidates_v2 against v1 on multiple topologies
    draft_tokens = [10, 20, 30, 40, 50]
    parent_indices = [-1, 0, 0, 1, 2]
    
    # Target logits:
    # Node 0 predicts 20 (selects 1)
    # Node 1 predicts 40 (selects 3)
    target_logits = np.zeros((5, 100))
    target_logits[0, 20] = 10.0
    target_logits[1, 40] = 10.0

    v1_path = verify_v1(draft_tokens, parent_indices, target_logits)
    v2_path = verify_tree_candidates_v2(draft_tokens, parent_indices, target_logits)

    assert v1_path == [10, 20, 40]
    assert v2_path == [10, 20, 40]
    assert v1_path == v2_path


def test_verify_tree_candidates_v2_with_context_and_stats():
    # Tree: 0 (root, token 10) -> 1 (child, token 20) -> 2 (child, token 30)
    draft_tokens = [10, 20, 30]
    parent_indices = [-1, 0, 1]
    target_logits = np.zeros((3, 50))
    target_logits[0, 20] = 5.0
    target_logits[1, 30] = 5.0

    # Context logits predict token 99, mismatching root token 10
    context_logits = np.zeros(50)
    context_logits[9] = 10.0  # correction token 9

    res_rejected, stats_rejected = verify_tree_candidates_v2(
        draft_tokens, parent_indices, target_logits, context_logits=context_logits, return_acceptance_stats=True
    )
    assert res_rejected == [9]
    assert stats_rejected["acceptance_rate"] == 0.0
    assert stats_rejected["accepted_count"] == 1
    assert stats_rejected["total_nodes"] == 3

    # Match context
    context_logits[10] = 15.0
    res_accepted, stats_accepted = verify_tree_candidates_v2(
        draft_tokens, parent_indices, target_logits, context_logits=context_logits, return_acceptance_stats=True
    )
    assert res_accepted == [10, 20, 30]
    assert pytest.approx(stats_accepted["acceptance_rate"]) == 1.0
    assert stats_accepted["accepted_count"] == 3
    assert stats_accepted["max_depth_accepted"] == 3


def test_verify_longest_path_when_multiple_branches_valid():
    # Tree:
    # 0 (tok 10) -> 1 (tok 20) -> 3 (tok 40) -> 5 (tok 60) [length 4]
    #            -> 2 (tok 30) -> 4 (tok 50)               [length 3]
    draft_tokens = [10, 20, 30, 40, 50, 60]
    parent_indices = [-1, 0, 0, 1, 2, 3]

    target_logits = np.zeros((6, 100))
    target_logits[0, 20] = 10.0 # 0 predicts 20 (branch 1)
    target_logits[1, 40] = 10.0 # 1 predicts 40 (branch 3)
    target_logits[3, 60] = 10.0 # 3 predicts 60 (branch 5)

    path, stats = verify_tree_candidates_v2(
        draft_tokens, parent_indices, target_logits, return_acceptance_stats=True
    )
    assert path == [10, 20, 40, 60]
    assert stats["max_depth_accepted"] == 4
    assert pytest.approx(stats["acceptance_rate"]) == 4 / 6


def test_empty_inputs():
    mask, ancestors = build_vectorized_tree_attention_mask([])
    assert mask.shape == (0, 0)
    assert ancestors.shape == (0, 0)

    res, stats = verify_tree_candidates_v2([], [], np.zeros((0, 10)), return_acceptance_stats=True)
    assert res == []
    assert stats["acceptance_rate"] == 0.0


def test_validate_tree_topology_cycles_and_bounds():
    validate_tree_topology([-1, 0, 1])
    gen = TreeSpeculativeGenerator(max_draft_tokens=64)
    assert gen.max_draft_tokens == 64
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_tree_topology([0])
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_tree_topology([1, 0])
    with pytest.raises(ValueError, match="out of bounds"):
        validate_tree_topology([-1, 5])
    with pytest.raises(ValueError, match="exceeds max_nodes"):
        validate_tree_topology([-1, 0, 1], max_nodes=2)
