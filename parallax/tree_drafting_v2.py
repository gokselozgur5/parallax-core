"""Speculative Eagle-2 Tree Drafting v2: dynamic branching, vectorized tree attention, and matrix-based path verification."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TreeDraftNodeV2:
    """Represents a node in the dynamic speculative tree."""
    node_id: int
    token_id: int
    parent_idx: int = -1
    depth: int = 0
    score: float = 0.0


@dataclass
class TreeDraftTopology:
    """Encapsulates vectorized tree topology structure and attention mask."""
    tokens: np.ndarray             # shape: (N,) int64
    parent_indices: np.ndarray     # shape: (N,) int64
    depths: np.ndarray             # shape: (N,) int64
    attention_mask: np.ndarray     # shape: (N, N) int64
    ancestor_matrix: np.ndarray    # shape: (N, N) bool, ancestor_matrix[i, j] is True if j is ancestor of i (including i)
    node_scores: Optional[np.ndarray] = None


def validate_tree_topology(
    parent_indices: Union[List[int], np.ndarray], max_nodes: Optional[int] = None
) -> None:
    """Validate tree topology for bounds, invalid parent indices, and cycles."""
    parents = np.asarray(parent_indices, dtype=np.int64)
    n = len(parents)
    if n == 0:
        return
    if max_nodes is not None and n > max_nodes:
        raise ValueError(f"Tree node count {n} exceeds max_nodes {max_nodes}")
    if np.any((parents < -1) | (parents >= n)):
        raise ValueError(f"Parent index out of bounds for tree of size {n}")
    state = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if state[i] != 0:
            continue
        curr, path = i, []
        while curr != -1:
            if state[curr] == 1:
                raise ValueError(f"Cycle detected in tree topology involving node {curr}")
            if state[curr] == 2:
                break
            state[curr] = 1
            path.append(curr)
            curr = int(parents[curr])
        for node in path:
            state[node] = 2


def build_vectorized_tree_attention_mask(
    parent_indices: Union[List[int], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized construction of causal tree attention mask and ancestor reachability matrix.
    
    Uses binary transitive closure / matrix powering or vectorized backward pointer chasing.
    For N nodes, constructs:
      - ancestor_matrix: (N, N) bool array where ancestor_matrix[i, j] = True if j is on path from root to i (inclusive)
      - attention_mask: (N, N) int64 array (0/1 or additive mask) representing causal visibility.
    """
    parents = np.asarray(parent_indices, dtype=np.int64)
    n = len(parents)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool)
    validate_tree_topology(parents)

    # Initialize identity (each node attends to itself)
    ancestors = np.eye(n, dtype=bool)

    # Topological propagation: since parent < i in standard tree order, or general DAG/tree with parent_indices
    # If nodes are topologically sorted (parents appear before children or index arbitrary):
    # Transitive closure on directed parent graph:
    # A directed edge exists from parent to child: adj[p, c] = 1.
    # In tree attention, node i attends to node j if j is an ancestor of i.
    # We can iteratively resolve or do topological / dynamic programming.
    for i in range(n):
        curr = parents[i]
        while curr != -1 and curr < n:
            ancestors[i, curr] = True
            curr = parents[curr]

    mask = ancestors.astype(np.int64)
    return mask, ancestors


def prune_tree_topology(
    tokens: Union[List[int], np.ndarray],
    parent_indices: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    max_nodes: int = 64,
    top_k_per_node: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prune tree branches based on cumulative probability / draft scores and top_k_per_node constraint.
    
    Args:
        tokens: Token IDs for each node.
        parent_indices: Parent index for each node (-1 for root).
        scores: Score/confidence for each node.
        max_nodes: Maximum total nodes to keep in the pruned tree.
        top_k_per_node: Maximum number of children allowed per parent node.
        
    Returns:
        pruned_tokens: np.ndarray of kept tokens.
        new_parent_indices: np.ndarray of updated parent indices in the new indexing.
        kept_indices: Original node indices that were retained.
    """
    tokens_arr = np.asarray(tokens, dtype=np.int64)
    parents_arr = np.asarray(parent_indices, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    n = len(tokens_arr)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    validate_tree_topology(parents_arr)

    # Group children by parent
    children_by_parent: Dict[int, List[int]] = {}
    for idx, p in enumerate(parents_arr):
        children_by_parent.setdefault(p, []).append(idx)

    # Apply top_k_per_node pruning if specified
    allowed_nodes = set()
    
    def collect_nodes(parent_idx: int):
        kids = children_by_parent.get(parent_idx, [])
        if top_k_per_node is not None and len(kids) > top_k_per_node:
            # Sort kids descending by score
            kids = sorted(kids, key=lambda k: scores_arr[k], reverse=True)[:top_k_per_node]
        for k in kids:
            allowed_nodes.add(k)
            collect_nodes(k)

    # Roots have parent -1
    roots = children_by_parent.get(-1, [])
    if top_k_per_node is not None and len(roots) > top_k_per_node:
        roots = sorted(roots, key=lambda k: scores_arr[k], reverse=True)[:top_k_per_node]
    for r in roots:
        allowed_nodes.add(r)
        collect_nodes(r)

    # If exceeding max_nodes, retain highest scoring nodes while preserving ancestor connectivity
    if len(allowed_nodes) > max_nodes:
        # Sort allowed nodes by score descending
        sorted_by_score = sorted(list(allowed_nodes), key=lambda x: scores_arr[x], reverse=True)
        retained: set[int] = set()
        for cand in sorted_by_score:
            if len(retained) >= max_nodes:
                break
            # Add cand and all its ancestors to retained
            curr = cand
            path = []
            while curr != -1 and curr in allowed_nodes:
                path.append(curr)
                curr = parents_arr[curr]
            if len(retained) + len([p for p in path if p not in retained]) <= max_nodes:
                retained.update(path)
        kept_indices = np.array(sorted(list(retained)), dtype=np.int64)
    else:
        kept_indices = np.array(sorted(list(allowed_nodes)), dtype=np.int64)

    if len(kept_indices) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    # Remap parent indices
    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    pruned_tokens = tokens_arr[kept_indices]
    new_parents = np.array(
        [old_to_new[parents_arr[old]] if parents_arr[old] in old_to_new else -1 for old in kept_indices],
        dtype=np.int64
    )

    return pruned_tokens, new_parents, kept_indices


def generate_dynamic_eagle_tree(
    root_token_id: int,
    depth: int = 4,
    top_k_per_node: int = 2,
    predict_fn = None,
) -> TreeDraftTopology:
    """Generate a speculative candidate tree dynamically given depth and top_k_per_node.
    
    If predict_fn is provided, it is invoked as predict_fn(parent_tokens, depth) -> List[(token_id, score)].
    Otherwise, generates synthetic deterministic branch structure for topology testing.
    """
    nodes: List[TreeDraftNodeV2] = []
    root = TreeDraftNodeV2(node_id=0, token_id=root_token_id, parent_idx=-1, depth=0, score=1.0)
    nodes.append(root)

    current_layer = [0]
    next_node_id = 1

    for d in range(1, depth + 1):
        next_layer = []
        for parent_node_id in current_layer:
            parent_node = nodes[parent_node_id]
            if predict_fn is not None:
                candidates = predict_fn(parent_node.token_id, d, top_k_per_node)
            else:
                # Default synthetic branching
                candidates = [
                    (parent_node.token_id * 10 + k, 1.0 / (k + 1))
                    for k in range(1, top_k_per_node + 1)
                ]
            for tok_id, sc in candidates[:top_k_per_node]:
                new_node = TreeDraftNodeV2(
                    node_id=next_node_id,
                    token_id=tok_id,
                    parent_idx=parent_node_id,
                    depth=d,
                    score=parent_node.score * sc
                )
                nodes.append(new_node)
                next_layer.append(next_node_id)
                next_node_id += 1
        current_layer = next_layer
        if not current_layer:
            break

    tokens = np.array([node.token_id for node in nodes], dtype=np.int64)
    parents = np.array([node.parent_idx for node in nodes], dtype=np.int64)
    depths = np.array([node.depth for node in nodes], dtype=np.int64)
    scores = np.array([node.score for node in nodes], dtype=np.float64)

    mask, ancestors = build_vectorized_tree_attention_mask(parents)

    return TreeDraftTopology(
        tokens=tokens,
        parent_indices=parents,
        depths=depths,
        attention_mask=mask,
        ancestor_matrix=ancestors,
        node_scores=scores
    )


def verify_tree_candidates_v2(
    draft_tokens: Union[List[int], np.ndarray],
    parent_indices: Union[List[int], np.ndarray],
    target_logits: np.ndarray,
    context_logits: Optional[np.ndarray] = None,
    return_acceptance_stats: bool = False,
    max_draft_tokens: int = 512,
    max_candidate_paths: int = 1024,
) -> Union[List[int], Tuple[List[int], Dict[str, float]]]:
    """Vectorized verification of speculative tree candidates selecting longest accepted path.
    
    Uses matrix-vector boolean formulation:
    1. Computes target predictions: target_preds = argmax(target_logits, axis=-1).
    2. Builds/vectorizes ancestor reachability matrix A (shape N, N) where A[i, j] = 1 if j is ancestor of i.
    3. Node validity condition:
       A node i is valid iff:
       - If i is root (parent == -1): context_choice is None or draft_tokens[i] == context_choice.
       - If i has parent p: parent p is valid AND target_preds[p] == draft_tokens[i].
       This means along the entire path from root to i, every child's token matches its parent's prediction.
    4. Matrix formulation:
       For every non-root node i, edge_valid[i] = (target_preds[parent[i]] == draft_tokens[i]).
       For root node r, root_valid[r] = (context_choice is None or draft_tokens[r] == context_choice).
       A node i is valid if and only if EVERY ancestor j on the path to i is valid!
       In matrix terms:
       valid_nodes_mask = (ancestor_matrix @ (~single_step_valid)) == 0.
       Path length to node i is path_lengths[i] = sum(ancestor_matrix[i, :]).
    5. The longest accepted path corresponds to argmax(valid_nodes_mask * path_lengths).
    6. Extracts the accepted tokens via ancestor indexing with zero loops over paths.

    Args:
        draft_tokens: Draft token IDs.
        parent_indices: Parent indices (-1 for root).
        target_logits: Target model logits for each draft node (N, Vocab).
        context_logits: Optional context logits for root validation.
        return_acceptance_stats: Whether to return acceptance rate and metrics dictionary.

    Returns:
        Longest accepted token sequence, and optionally acceptance stats.
    """
    tokens = np.asarray(draft_tokens, dtype=np.int64)
    parents = np.asarray(parent_indices, dtype=np.int64)
    n = len(tokens)
    if n != len(parents):
        raise ValueError(f"Length mismatch: {n} tokens vs {len(parents)} parents")
    if n > max_draft_tokens:
        raise ValueError(f"Draft token length {n} exceeds max_draft_tokens {max_draft_tokens}")
    if n == 0:
        if return_acceptance_stats:
            return [], {"acceptance_rate": 0.0, "accepted_count": 0, "total_nodes": 0}
        return []
    if target_logits.shape[0] != n:
        raise ValueError(f"target_logits shape {target_logits.shape} mismatch with {n} tokens")
    validate_tree_topology(parents, max_nodes=max_draft_tokens)
    leaves = sum(1 for i in range(n) if i not in set(parents))
    if leaves > max_candidate_paths:
        raise ValueError(f"Candidate paths {leaves} exceeds limit {max_candidate_paths}")

    target_preds = np.argmax(target_logits, axis=-1)
    context_choice = int(np.argmax(context_logits)) if context_logits is not None else None

    # Step 1: Compute single-step validity for each node
    # Root condition: parent == -1
    is_root = (parents == -1)
    single_valid = np.zeros(n, dtype=bool)

    # For roots:
    if context_choice is None:
        single_valid[is_root] = True
    else:
        single_valid[is_root] = (tokens[is_root] == context_choice)

    # For non-roots: valid iff target predicted draft_token from parent
    non_roots = ~is_root
    if np.any(non_roots):
        parent_preds = target_preds[parents[non_roots]]
        single_valid[non_roots] = (tokens[non_roots] == parent_preds)

    # Step 2: Ancestor matrix A of shape (N, N), bool
    # A[i, j] is True if j is an ancestor of i (including i)
    _, ancestor_matrix = build_vectorized_tree_attention_mask(parents)

    # Step 3: A node i is globally valid if and only if ALL its ancestors are single_valid.
    # That is: for all j where ancestor_matrix[i, j] is True, single_valid[j] is True.
    # In vector logic: invalid_ancestor = ancestor_matrix @ (~single_valid)
    # A node is valid iff invalid_ancestor[i] == 0.
    invalid_ancestor_count = ancestor_matrix.dot((~single_valid).astype(np.int32))
    node_validity = (invalid_ancestor_count == 0)

    # Step 4: Path lengths = number of ancestors for each node
    path_lengths = np.sum(ancestor_matrix, axis=1)  # shape (N,)
    valid_lengths = np.where(node_validity, path_lengths, 0)

    max_len = np.max(valid_lengths) if len(valid_lengths) > 0 else 0

    if max_len == 0:
        # All roots/candidates rejected
        res = [context_choice] if context_choice is not None else []
        if return_acceptance_stats:
            return res, {
                "acceptance_rate": 0.0,
                "accepted_count": len(res),
                "total_nodes": n,
            }
        return res

    # Select node with maximum valid path length
    best_leaf = int(np.argmax(valid_lengths))

    # Step 5: Extract sequence of tokens for best_leaf
    # Ancestors of best_leaf: boolean mask ancestor_matrix[best_leaf]
    leaf_ancestors = ancestor_matrix[best_leaf]
    ancestor_indices = np.where(leaf_ancestors)[0]
    
    # Sort ancestor indices by their depth (path_lengths) to get proper root-to-leaf ordering
    order = np.argsort(path_lengths[ancestor_indices])
    accepted_path = tokens[ancestor_indices[order]].tolist()

    if return_acceptance_stats:
        acc_rate = float(len(accepted_path) / n) if n > 0 else 0.0
        stats = {
            "acceptance_rate": acc_rate,
            "accepted_count": len(accepted_path),
            "total_nodes": n,
            "max_depth_accepted": int(max_len),
        }
        return accepted_path, stats

    return accepted_path


class TreeSpeculativeGenerator:
    """Speculative tree generator and verification coordinator."""

    def __init__(
        self,
        max_draft_tokens: int = 64,
        max_sequence_length: int = 2048,
        max_candidate_paths: int = 128,
        max_successors_per_node: int = 16,
    ) -> None:
        if max_draft_tokens <= 0 or max_sequence_length <= 0:
            raise ValueError("Token bounds must be positive")
        if max_candidate_paths <= 0 or max_successors_per_node <= 0:
            raise ValueError("Candidate and successor bounds must be positive")
        self.max_draft_tokens = max_draft_tokens
        self.max_sequence_length = max_sequence_length
        self.max_candidate_paths = max_candidate_paths
        self.max_successors_per_node = max_successors_per_node

    def validate_configuration(
        self,
        draft_tokens: Union[List[int], np.ndarray],
        parent_indices: Union[List[int], np.ndarray],
        current_sequence_length: int = 0,
    ) -> None:
        """Validate token length, sequence limits, graph cycles, and successors."""
        tokens, parents = np.asarray(draft_tokens, dtype=np.int64), np.asarray(parent_indices, dtype=np.int64)
        n = len(tokens)
        if n != len(parents):
            raise ValueError(f"Length mismatch: {n} tokens vs {len(parents)} parents")
        if n > self.max_draft_tokens:
            raise ValueError(f"Draft token length {n} exceeds max_draft_tokens {self.max_draft_tokens}")
        if current_sequence_length + n > self.max_sequence_length:
            raise ValueError(f"Sequence length {current_sequence_length + n} exceeds {self.max_sequence_length}")
        validate_tree_topology(parents, max_nodes=self.max_draft_tokens)
        if n > 0:
            counts: Dict[int, int] = {}
            for p in parents:
                counts[p] = counts.get(p, 0) + 1
                if p != -1 and counts[p] > self.max_successors_per_node:
                    raise ValueError(f"Parent {p} exceeds max successors {self.max_successors_per_node}")
            leaves = sum(1 for i in range(n) if i not in set(parents))
            if leaves > self.max_candidate_paths:
                raise ValueError(f"Candidate paths {leaves} exceeds limit {self.max_candidate_paths}")

    def verify(
        self,
        draft_tokens: Union[List[int], np.ndarray],
        parent_indices: Union[List[int], np.ndarray],
        target_logits: np.ndarray,
        context_logits: Optional[np.ndarray] = None,
        return_acceptance_stats: bool = False,
        current_sequence_length: int = 0,
    ) -> Union[List[int], Tuple[List[int], Dict[str, float]]]:
        """Validate topology and verify candidates against target logits."""
        self.validate_configuration(draft_tokens, parent_indices, current_sequence_length)
        return verify_tree_candidates_v2(
            draft_tokens, parent_indices, target_logits,
            context_logits=context_logits,
            return_acceptance_stats=return_acceptance_stats,
            max_draft_tokens=self.max_draft_tokens,
            max_candidate_paths=self.max_candidate_paths,
        )

    def generate(
        self, root_token_id: int, depth: int = 4, top_k_per_node: int = 2, predict_fn=None
    ) -> TreeDraftTopology:
        """Generate speculative tree enforcing node and successor bounds."""
        if top_k_per_node > self.max_successors_per_node:
            raise ValueError(f"top_k_per_node {top_k_per_node} exceeds {self.max_successors_per_node}")
        topo = generate_dynamic_eagle_tree(root_token_id, depth=depth, top_k_per_node=top_k_per_node, predict_fn=predict_fn)
        self.validate_configuration(topo.tokens, topo.parent_indices)
        return topo
