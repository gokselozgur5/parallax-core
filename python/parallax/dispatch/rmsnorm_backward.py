"""RMSNorm backward autograd functional kernel with dispatch support."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon ensuring strictly positive bounded threshold."""
    return max(float(eps), min_eps)
