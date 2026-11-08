"""Unified Normalization (LayerNorm & RMSNorm) with fused residuals."""

from typing import Optional, Tuple
import numpy as np


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
            rms = np.sqrt(np.mean(fused_x ** 2, axis=-1, keepdims=True) + self.eps)
            normed = (fused_x / rms) * self.weight
        else:
            mean = np.mean(fused_x, axis=-1, keepdims=True)
            var = np.var(fused_x, axis=-1, keepdims=True)
            normed = ((fused_x - mean) / np.sqrt(var + self.eps)) * self.weight + self.bias
        return normed, fused_x
