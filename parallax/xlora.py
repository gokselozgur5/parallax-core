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


    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with dynamic gating across adapters.

        Args:
            x: Input activation of shape (batch, in_features) or (in_features,).

        Returns:
            Tuple of (output, gating_weights).
        """
        x_in = x if x.ndim == 2 else x[np.newaxis, :]
        # Base projection: x @ W_base.T
        base_out = np.dot(x_in, self.base_weight.T)

        # Dynamic router logits: x @ gate_weight -> (B, num_adapters)
        gate_logits = np.dot(x_in, self.gate_weight)
        # Softmax gating weights
        gate_m = np.max(gate_logits, axis=-1, keepdims=True)
        exp_gates = np.exp(gate_logits - gate_m)
        gate_weights = exp_gates / np.sum(exp_gates, axis=-1, keepdims=True)

        # Dynamic adapter outputs: sum_i g_i * (x @ A_i.T @ B_i.T * scaling)
        adapter_out = np.zeros_like(base_out)
        for i in range(self.num_adapters):
            g_i = gate_weights[:, i : i + 1]  # (B, 1)
            # x @ A_i^T: (B, rank)
            a_out = np.dot(x_in, self.lora_A[i].T)
            # a_out @ B_i^T: (B, out_features)
            b_out = np.dot(a_out, self.lora_B[i].T) * self.scaling
            adapter_out += g_i * b_out

        total_out = base_out + adapter_out
        if x.ndim == 1:
            return total_out.squeeze(0), gate_weights.squeeze(0)
        return total_out, gate_weights
