"""Medusa multi-head speculative decoding without draft models.

Reference: Cai et al., 2023 - Medusa: Simple Framework for Accelerating LLM
Generation with Multiple Decoding Heads.
Adds extra linear heads to predict multiple future tokens simultaneously,
verifying candidate continuations via tree/prefix matching.
"""

from typing import List, Tuple
import numpy as np


class MedusaHead:
    """A single Medusa linear projection head predicting token at offset +k."""

    def __init__(self, hidden_dim: int, vocab_size: int, offset: int):
        self.offset = offset
        limit = np.sqrt(6.0 / (hidden_dim + vocab_size))
        self.weight = np.random.uniform(-limit, limit, (hidden_dim, vocab_size)).astype(np.float32)
        self.bias = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, hidden_state: np.ndarray) -> np.ndarray:
        """Projects hidden state (d,) or (B, d) to vocabulary logits (vocab_size,)."""
        return np.dot(hidden_state, self.weight) + self.bias



class MedusaModel:
    """Wrapper holding multiple Medusa heads for speculative prediction."""

    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 3):
        self.num_heads = num_heads
        self.heads = [
            MedusaHead(hidden_dim, vocab_size, offset=i + 1)
            for i in range(num_heads)
        ]

    def predict_candidates(self, hidden_state: np.ndarray) -> List[int]:
        """Predict top-1 token for each Medusa head offset."""
        candidates = []
        for head in self.heads:
            logits = head.forward(hidden_state)
            token = int(np.argmax(logits))
            candidates.append(token)
        return candidates


def verify_medusa_candidates(
    candidates: List[int],
    ground_truth_fn,
    prefix_tokens: List[int],
) -> Tuple[List[int], int]:
    """Greedy prefix verification of Medusa speculative candidates.

    Args:
        candidates: List of proposed candidate tokens from Medusa heads.
        ground_truth_fn: Callable taking list of tokens and returning next true token.
        prefix_tokens: Current sequence of tokens.

    Returns:
        Tuple of (accepted_tokens, next_token):
            - accepted_tokens: List of candidate tokens that matched ground truth.
            - next_token: First divergence token or subsequent true token.
    """
    accepted = []
    current_seq = list(prefix_tokens)

    for cand in candidates:
        true_next = ground_truth_fn(current_seq)
        if cand == true_next:
            accepted.append(cand)
            current_seq.append(cand)
        else:
            return accepted, true_next

    # If all candidates accepted, sample one more token from base model
    extra_token = ground_truth_fn(current_seq)
    return accepted, extra_token
