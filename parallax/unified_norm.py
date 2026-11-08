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
