"""LayerNorm backward autograd functional kernel with dispatch support."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)
