"""Unified Normalization (LayerNorm & RMSNorm) with fused residuals."""

from typing import Optional, Tuple
import numpy as np

from parallax.c_engine import is_c_engine_available, c_rmsnorm


class UnifiedNorm:
    """Supports both LayerNorm and RMSNorm with fused residual addition."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        norm_type: str = "rmsnorm",
    ) -> None:
        self.dim = int(dim)
        self.eps = float(eps)
        self.norm_type = str(norm_type).lower()
        self.weight = np.ones(dim, dtype=np.float64)
        self.bias = np.zeros(dim, dtype=np.float64) if self.norm_type == "layernorm" else None

    def forward(
        self,
        x: np.ndarray,
        residual: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalization forward with optional fused residual addition."""
        fused_x = x + residual if residual is not None else x
        if self.norm_type == "rmsnorm":
            if is_c_engine_available():
                normed = c_rmsnorm(fused_x, self.weight, eps=self.eps)
            else:
                rms = np.sqrt(np.mean(fused_x ** 2, axis=-1, keepdims=True) + self.eps)
                normed = (fused_x / rms) * self.weight
        else:
            mean = np.mean(fused_x, axis=-1, keepdims=True)
            var = np.var(fused_x, axis=-1, keepdims=True)
            normed = ((fused_x - mean) / np.sqrt(var + self.eps)) * self.weight
            if self.bias is not None:
                normed = normed + self.bias
        return normed, fused_x

    def backward(
        self,
        grad_out: np.ndarray,
        fused_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Analytical backward pass returning (grad_fused_x, grad_weight, grad_bias)."""
        d_hat = grad_out * self.weight
        if self.norm_type == "rmsnorm":
            rms = np.sqrt(np.mean(fused_x ** 2, axis=-1, keepdims=True) + self.eps)
            x_hat = fused_x / rms
            grad_x = (1.0 / rms) * (d_hat - x_hat * np.mean(d_hat * x_hat, axis=-1, keepdims=True))
            grad_w = np.sum(grad_out * x_hat, axis=tuple(range(grad_out.ndim - 1)))
            return grad_x, grad_w, None
        else:
            mean = np.mean(fused_x, axis=-1, keepdims=True)
            var = np.var(fused_x, axis=-1, keepdims=True)
            std_inv = 1.0 / np.sqrt(var + self.eps)
            x_hat = (fused_x - mean) * std_inv
            grad_x = std_inv * (d_hat - np.mean(d_hat, axis=-1, keepdims=True) - x_hat * np.mean(d_hat * x_hat, axis=-1, keepdims=True))
            grad_w = np.sum(grad_out * x_hat, axis=tuple(range(grad_out.ndim - 1)))
            grad_b = np.sum(grad_out, axis=tuple(range(grad_out.ndim - 1)))
            return grad_x, grad_w, grad_b
