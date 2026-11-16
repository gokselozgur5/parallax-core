"""Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np


def stabilize_epsilon(eps: float, min_eps: float = 1e-12) -> float:
    """Stabilize epsilon parameter ensuring positive strictly-bounded value."""
    eps_val = float(eps)
    return max(eps_val, min_eps)
