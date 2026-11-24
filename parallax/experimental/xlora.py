"""X-LoRA: Mixture of Low-Rank Adapters with dynamic gating.

Reference: Buehler et al., 2024 - X-LoRA: Mixture of Low-Rank Adapter Experts,
A Dense Gating Approach for Fine-Tuned LLMs.
Blends multiple specialized LoRA adapters via an input-dependent gating network.

Integration contract:
    Standalone mathematical kernel. Operates on pure NumPy arrays and maintains
    explicit manual gradients via backward(). Not a subclass of parallax.nn.Module
    or integrated into the computational autograd graph.
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


    def backward(
        self,
        x: np.ndarray,
        grad_out: np.ndarray,
        gate_weights: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """Compute parameter gradients for active LoRA adapters A_i and B_i,

        along with gate weight gradient and input gradient dL/dx.

        Args:
            x: Input activation of shape (B, in_features) or (in_features,).
            grad_out: Output gradient dL/dY of shape (B, out_features) or (out_features,).
            gate_weights: Softmax router probabilities of shape (B, num_adapters) or (num_adapters,).

        Returns:
            Tuple of (grads_A, grads_B, d_gate, dx).
        """
        x_in = x if x.ndim == 2 else x[np.newaxis, :]
        grad_in = grad_out if grad_out.ndim == 2 else grad_out[np.newaxis, :]
        gates = gate_weights if gate_weights.ndim == 2 else gate_weights[np.newaxis, :]

        grads_A = []
        grads_B = []

        d_g = np.zeros_like(gates)
        dx_adapters = np.zeros_like(x_in)

        for i in range(self.num_adapters):
            g_i = gates[:, i : i + 1]  # (B, 1)
            # Weighted output gradient: g_i * grad_in * scaling
            eff_grad = g_i * grad_in * self.scaling

            # a_out = x @ A_i^T: (B, rank)
            a_out = np.dot(x_in, self.lora_A[i].T)

            # grad_B = eff_grad^T @ a_out: (out_features, rank)
            grad_B = np.dot(eff_grad.T, a_out)

            # grad_a = eff_grad @ B_i: (B, rank)
            grad_a = np.dot(eff_grad, self.lora_B[i])

            # grad_A = grad_a^T @ x_in: (rank, in_features)
            grad_A = np.dot(grad_a.T, x_in)

            grads_A.append(grad_A)
            grads_B.append(grad_B)

            # Adapter output contribution to router gradient
            adapter_out_i = np.dot(a_out, self.lora_B[i].T) * self.scaling
            d_g[:, i] = np.sum(grad_in * adapter_out_i, axis=-1)

            # Input gradient contribution through adapter i
            dx_adapters += np.dot(grad_a, self.lora_A[i])

        # Softmax backward for gate weights:
        # d_logits = gates * (d_g - sum(d_g * gates, axis=-1, keepdims=True))
        d_logits = gates * (d_g - np.sum(d_g * gates, axis=-1, keepdims=True))
        d_gate = np.dot(x_in.T, d_logits)

        # Full input gradient: base layer + adapters + gating network
        dx_base = np.dot(grad_in, self.base_weight)
        dx_gate = np.dot(d_logits, self.gate_weight.T)
        dx = dx_base + dx_adapters + dx_gate

        if x.ndim == 1:
            dx = dx.squeeze(0)

        return grads_A, grads_B, d_gate, dx
