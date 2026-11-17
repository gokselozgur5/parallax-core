"""Fused GeLU backward functional kernels with tanh derivative support."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

SQRT_2_OVER_PI: float = 0.7978845608028654
COEF: float = 0.044715
SQRT_2: float = 1.4142135623730951
SQRT_2_PI: float = 2.5066282746310002
