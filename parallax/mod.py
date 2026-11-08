"""Mixture-of-Depths (MoD) dynamic compute allocation.

Reference: Raposo et al., DeepMind 2024 - Mixture-of-Depths: Dynamically allocating
compute in transformer-based language models.
Routes top-k tokens based on router scores through the transformer block while
inactive tokens bypass via residual connection.
"""

from typing import Tuple, Callable
import numpy as np


class MoDRouter:
    """Computes scalar token weights and selects top-k tokens within capacity budget."""

    def __init__(self, hidden_dim: int, capacity_factor: float = 0.5):
        self.hidden_dim = hidden_dim
        self.capacity_factor = capacity_factor
        self.weight = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.02


    def route_tokens(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute token router scores and select top-capacity token indices.

        Args:
            x: Input sequence activations of shape (seq_len, hidden_dim).

        Returns:
            Tuple of:
                - selected_indices: Array of top-capacity token positions.
                - router_weights: Gating scores for selected tokens, shape (C, 1).
                - all_scores: Raw sigmoid scores for all tokens, shape (seq_len,).
        """
        seq_len = x.shape[0]
        # Raw projection to router scalar logits
        logits = np.dot(x, self.weight).squeeze(-1)  # (seq_len,)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        # Capacity C = max(1, int(capacity_factor * seq_len))
        capacity = max(1, int(np.ceil(self.capacity_factor * seq_len)))

        # Pick top-C token indices (sorted to preserve causality / temporal order)
        top_indices = np.argsort(-scores)[:capacity]
        sorted_indices = np.sort(top_indices)

        selected_weights = scores[sorted_indices, np.newaxis]
        return sorted_indices, selected_weights, scores
