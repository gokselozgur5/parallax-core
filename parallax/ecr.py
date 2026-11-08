"""Expert Choice Routing (ECR) for load-balanced Mixture of Experts."""

from typing import List, Tuple
import numpy as np


class ExpertChoiceRouter:
    """Routes tokens where experts select top tokens ensuring perfect balance."""

    def __init__(self, num_experts: int, capacity_factor: float = 1.0) -> None:
        self.num_experts = int(num_experts)
        self.capacity_factor = float(capacity_factor)

    def compute_expert_capacity(self, num_tokens: int) -> int:
        """Calculate uniform token capacity per expert."""
        cap = int(np.ceil((num_tokens * self.capacity_factor) / self.num_experts))
        return max(1, cap)

    def route_tokens(
        self, gating_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Each expert independently selects its top-k tokens based on scores."""
        num_tokens, num_experts = gating_scores.shape
        capacity = self.compute_expert_capacity(num_tokens)
        expert_token_indices = np.zeros((num_experts, capacity), dtype=int)
        expert_gate_weights = np.zeros((num_experts, capacity), dtype=float)

        for e in range(num_experts):
            col_scores = gating_scores[:, e]
            topk_idx = np.argsort(col_scores)[::-1][:capacity]
            expert_token_indices[e] = topk_idx
            expert_gate_weights[e] = col_scores[topk_idx]
        return expert_token_indices, expert_gate_weights
