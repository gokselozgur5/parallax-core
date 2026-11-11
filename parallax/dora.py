"""Weight-Decomposed Low-Rank Adaptation (DoRA).

Reference: Liu et al., 2024 - DoRA: Weight-Decomposed Low-Rank Adaptation.
Decomposes pretrained weights into magnitude and directional components, updating
the directional component via LoRA while training magnitude separately.

Integration contract:
    Standalone mathematical kernel. Operates on pure NumPy arrays and does not
    subclass parallax.nn.Module or register Parameters in the autograd DAG.
"""

from typing import Optional, Tuple
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


from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


class DoRAModule(Module):
    """Autograd-integrated Module adapter for DoRA fine-tuning."""

    def __init__(
        self,
        base_layer: Linear,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        self.base_layer.weight.grad = None
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            self.base_layer.bias.grad = None

        # Base direction matrix V
        v_data = self.base_layer.weight.data.copy().astype(np.float32)
        # Learnable magnitude vector m = ||V||_col
        m_data = np.linalg.norm(v_data, axis=1).astype(np.float32)

        self.m = Parameter(m_data, label=f"DoRA_m_{self.out_features}")
        self.base_V = v_data

        self.lora_A: Optional[Parameter] = None
        self.lora_B: Optional[Parameter] = None
        if rank > 0:
            limit_a = 1.0 / np.sqrt(self.in_features)
            a_data = np.random.normal(0.0, limit_a, (rank, self.in_features)).astype(np.float32)
            b_data = np.zeros((self.out_features, rank), dtype=np.float32)
            self.lora_A = Parameter(a_data, label=f"DoRA_A_{self.in_features}x{rank}")
            self.lora_B = Parameter(b_data, label=f"DoRA_B_{rank}x{self.out_features}")

    def forward(self, x: Tensor) -> Tensor:
        if self.rank == 0 or self.lora_A is None or self.lora_B is None:
            return self.base_layer(x)

        x_data = x.data
        was_1d = (x_data.ndim == 1)
        x_in = x_data if not was_1d else x_data[np.newaxis, :]

        delta_V = self.scaling * np.dot(self.lora_B.data, self.lora_A.data)
        V_total = self.base_V + delta_V
        norms = np.linalg.norm(V_total, axis=1, keepdims=True) + 1e-8
        V_norm = V_total / norms
        m_col = self.m.data[:, np.newaxis]
        W_eff = m_col * V_norm

        out_data = np.dot(x_in, W_eff.T)
        if self.base_layer.bias is not None:
            out_data += self.base_layer.bias.data
        if was_1d:
            out_data = out_data.squeeze(0)

        out = Tensor(out_data, (x, self.m, self.lora_A, self.lora_B), "DoRAModule")
        if not out.requires_grad:
            return out

        def _backward() -> None:
            if out.grad is None:
                return
            grad_y = out.grad if not was_1d else out.grad[np.newaxis, :]
            grad_y_2d = grad_y.reshape(-1, self.out_features)
            x_in_2d = x_in.reshape(-1, self.in_features)

            grad_W_eff = np.dot(grad_y_2d.T, x_in_2d)

            if self.m.requires_grad:
                grad_m = np.sum(grad_W_eff * V_norm, axis=1)
                _accumulate_grad(self.m, grad_m)

            proj = np.sum(grad_W_eff * V_total, axis=1, keepdims=True) / (norms**2)
            grad_V = (m_col / norms) * (grad_W_eff - proj * V_total)

            if self.lora_B is not None and self.lora_A is not None:
                if self.lora_B.requires_grad:
                    grad_B = self.scaling * np.dot(grad_V, self.lora_A.data.T)
                    _accumulate_grad(self.lora_B, grad_B)

                if self.lora_A.requires_grad:
                    grad_A = self.scaling * np.dot(self.lora_B.data.T, grad_V)
                    _accumulate_grad(self.lora_A, grad_A)

            if x.requires_grad:
                dx = np.dot(grad_y_2d, W_eff)
                _accumulate_grad(x, dx.reshape(x_data.shape))

        out._backward = _backward
        return out
