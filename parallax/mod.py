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

    def __init__(self, hidden_dim: int, capacity_factor: float = 0.5, threshold: float = 0.5):
        self.hidden_dim = hidden_dim
        self.capacity_factor = capacity_factor
        self.threshold = threshold
        self.weight = np.random.randn(hidden_dim, 1).astype(np.float32) * 0.02


    def route_tokens(
        self,
        x: np.ndarray,
        causal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute token router scores and select top-capacity token indices.

        In training / non-causal mode (causal=False), top-C tokens are selected across
        the entire sequence.
        In causal inference/decoding mode (causal=True), routing for token t must not
        depend on future tokens > t. Tokens are selected either via per-token threshold
        routing (score >= threshold) or causally clamped prefix capacity so future
        context does not leak into routing decisions at position t.

        Args:
            x: Input sequence activations of shape (seq_len, hidden_dim).
            causal: Whether to enforce causal routing for autoregressive decoding.

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

        if causal:
            # Causal per-token threshold routing: token t depends only on scores[t] <= t
            selected_mask = scores >= self.threshold
            selected_indices = np.where(selected_mask)[0]
            if len(selected_indices) == 0:
                # Fallback to token with highest score so far to guarantee at least 1 token
                selected_indices = np.array([int(np.argmax(scores))], dtype=int)
            selected_weights = scores[selected_indices, np.newaxis]
            return selected_indices, selected_weights, scores

        # Training / non-causal sequence routing:
        # Capacity C = max(1, int(capacity_factor * seq_len))
        capacity = max(1, int(np.ceil(self.capacity_factor * seq_len)))

        # Pick top-C token indices (sorted to preserve causality / temporal order)
        top_indices = np.argsort(-scores)[:capacity]
        sorted_indices = np.sort(top_indices)

        selected_weights = scores[sorted_indices, np.newaxis]
        return sorted_indices, selected_weights, scores



def mod_block_forward(
    x: np.ndarray,
    router: MoDRouter,
    block_fn: Callable[[np.ndarray], np.ndarray],
    causal: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute Mixture-of-Depths forward pass.

    Args:
        x: Input activations of shape (seq_len, hidden_dim).
        router: MoDRouter instance.
        block_fn: Transformer computation block taking selected tokens (C, d) and returning (C, d).
        causal: Whether to use causal routing for autoregressive inference.

    Returns:
        Tuple of:
            - output: Full sequence activations of shape (seq_len, hidden_dim).
            - selected_indices: Array of token indices that were routed through block_fn.
    """
    selected_indices, router_weights, _ = router.route_tokens(x, causal=causal)

    # Gather selected tokens
    x_selected = x[selected_indices]

    # Compute heavy transformer block only on selected capacity tokens
    block_out = block_fn(x_selected)

    # Weight by router score and apply residual connection
    weighted_block_out = x_selected + router_weights * block_out

    # Scatter back into output sequence (unselected tokens pass through purely via identity)
    output = np.copy(x)
    output[selected_indices] = weighted_block_out

    return output, selected_indices
