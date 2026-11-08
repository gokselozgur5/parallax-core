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
