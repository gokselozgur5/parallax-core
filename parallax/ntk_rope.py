"""NTK-aware Scaled Rotary Position Embedding (RoPE)."""

from typing import Tuple
import numpy as np


class NTKAwareRoPE:
    """Computes modulated RoPE frequencies using NTK-aware base scaling."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        alpha: float = 1.0,
    ) -> None:
        self.dim = int(dim)
        self.base = float(base)
        self.alpha = float(alpha)
