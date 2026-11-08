"""X-LoRA: Mixture of Low-Rank Adapters with dynamic gating.

Reference: Buehler et al., 2024 - X-LoRA: Mixture of Low-Rank Adapter Experts,
A Dense Gating Approach for Fine-Tuned LLMs.
Blends multiple specialized LoRA adapters via an input-dependent gating network.
"""

from typing import List, Tuple
import numpy as np


class XLoRALinear:
    """Linear layer augmented with multiple dynamically routed LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_adapters: int = 4,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_adapters = num_adapters
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen base weights
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.base_weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)

        # Multiple LoRA adapters: A_i ~ Gaussian, B_i = 0
        self.lora_A = [
            np.random.randn(rank, in_features).astype(np.float32) * 0.01
            for _ in range(num_adapters)
        ]
        self.lora_B = [
            np.zeros((out_features, rank), dtype=np.float32)
            for _ in range(num_adapters)
        ]

        # Gating network: projects input to router logits across adapters
        self.gate_weight = np.random.randn(in_features, num_adapters).astype(np.float32) * 0.01
