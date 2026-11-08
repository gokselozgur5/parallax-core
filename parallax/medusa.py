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
