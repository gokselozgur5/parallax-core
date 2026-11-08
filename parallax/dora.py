"""Weight-Decomposed Low-Rank Adaptation (DoRA).

Reference: Liu et al., 2024 - DoRA: Weight-Decomposed Low-Rank Adaptation.
Decomposes pretrained weights into magnitude and directional components, updating
the directional component via LoRA while training magnitude separately.
"""

from typing import Tuple
import numpy as np


class DoRALinear:
    """Linear layer using Weight-Decomposed Low-Rank Adaptation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Pretrained weight V (frozen base direction)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.base_V = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)

        # Learnable magnitude vector m = ||base_V||_col (shape: out_features)
        self.m = np.linalg.norm(self.base_V, axis=1).astype(np.float32)

        # Low-rank directional updates: A ~ Gaussian, B = 0
        self.lora_A = np.random.randn(rank, in_features).astype(np.float32) * 0.01
        self.lora_B = np.zeros((out_features, rank), dtype=np.float32)
