"""DispatchedConv2d module routing 2D convolution through backend dispatch."""

from __future__ import annotations

from typing import Any, Tuple, Union
import numpy as np

from parallax.backends.dispatch import dispatch
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


def _pair(val: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert int or tuple to a 2-tuple of ints."""
    if isinstance(val, (int, np.integer)):
        return (int(val), int(val))
    if isinstance(val, (tuple, list)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    raise ValueError(f"Expected int or pair of ints, got {val}")
