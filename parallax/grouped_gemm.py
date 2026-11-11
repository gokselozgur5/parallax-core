"""Grouped GEMM kernel for contiguous expert dispatch."""

from typing import List, Tuple
import numpy as np


class GroupedGEMMDispatcher:
    """Manages contiguous activation grouping and GEMM dispatch across MoE experts."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = int(num_experts)

    def group_tokens(
        self,
        tokens: np.ndarray,
        expert_assignments: List[int],
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Group tokens by their assigned expert ID into contiguous batches."""
        expert_batches: List[List[np.ndarray]] = [[] for _ in range(self.num_experts)]
        expert_orig_indices: List[List[int]] = [[] for _ in range(self.num_experts)]

        for idx, (token, exp_id) in enumerate(zip(tokens, expert_assignments)):
            expert_batches[exp_id].append(token)
            expert_orig_indices[exp_id].append(idx)

        stacked_batches = [
            np.array(b) if len(b) > 0 else np.zeros((0, tokens.shape[-1]))
            for b in expert_batches
        ]
        return stacked_batches, expert_orig_indices

    def grouped_gemm_forward(
        self,
        tokens: np.ndarray,
        expert_assignments: List[int],
        expert_weights: List[np.ndarray],
    ) -> np.ndarray:
        """Execute grouped GEMM and restore token outputs to original sequence order."""
        batches, orig_indices = self.group_tokens(tokens, expert_assignments)
        out_dim = expert_weights[0].shape[-1]
        out = np.zeros((len(tokens), out_dim), dtype=float)

        for e in range(self.num_experts):
            if len(batches[e]) > 0:
                exp_out = np.matmul(batches[e], expert_weights[e])
                for token_out, orig_idx in zip(exp_out, orig_indices[e]):
                    out[orig_idx] = token_out
        return out
