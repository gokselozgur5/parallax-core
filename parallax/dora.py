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


    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using decomposed normalized directional matrix scaled by magnitude.

        W' = m * (V + scaling * B @ A) / ||V + scaling * B @ A||
        Y = X @ W'^T
        """
        # Directional matrix: V_total = V + scaling * (B @ A)
        delta_V = self.scaling * np.dot(self.lora_B, self.lora_A)
        V_total = self.base_V + delta_V

        # Column/row norms: norm over in_features axis (axis=1)
        norms = np.linalg.norm(V_total, axis=1, keepdims=True) + 1e-8
        V_norm = V_total / norms

        # Effective weight: m (out_features, 1) * V_norm
        W_eff = self.m[:, np.newaxis] * V_norm

        return np.dot(x, W_eff.T)


    def backward(
        self,
        x: np.ndarray,
        grad_out: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients for magnitude vector m, adapter matrices A and B, and input x.

        Args:
            x: Input activation of shape (B, in_features) or (in_features,).
            grad_out: Output gradient of shape (B, out_features) or (out_features,).

        Returns:
            Tuple of (grad_m, grad_A, grad_B, dx).
        """
        x_in = x if x.ndim == 2 else x[np.newaxis, :]
        grad_y = grad_out if grad_out.ndim == 2 else grad_out[np.newaxis, :]

        # grad_W_eff = grad_y^T @ x_in: shape (out_features, in_features)
        grad_W_eff = np.dot(grad_y.T, x_in)

        delta_V = self.scaling * np.dot(self.lora_B, self.lora_A)
        V_total = self.base_V + delta_V
        norms = np.linalg.norm(V_total, axis=1, keepdims=True) + 1e-8
        V_norm = V_total / norms

        # grad_m = sum_j (grad_W_eff * V_norm): shape (out_features,)
        grad_m = np.sum(grad_W_eff * V_norm, axis=1)

        # grad_V_total using quotient rule:
        # W_eff = m * V / ||V|| -> dW = m/||V|| * (I - V V^T / ||V||^2)
        m_col = self.m[:, np.newaxis]
        proj = np.sum(grad_W_eff * V_total, axis=1, keepdims=True) / (norms ** 2)
        grad_V = (m_col / norms) * (grad_W_eff - proj * V_total)

        # grad_delta_V = grad_V
        grad_B = self.scaling * np.dot(grad_V, self.lora_A.T)
        grad_A = self.scaling * np.dot(self.lora_B.T, grad_V)

        # Input gradient dL/dx = grad_y @ W_eff
        W_eff = m_col * V_norm
        dx = np.dot(grad_y, W_eff)
        if x.ndim == 1:
            dx = dx.squeeze(0)

        return grad_m, grad_A, grad_B, dx
