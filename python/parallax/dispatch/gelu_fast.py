"""GeLU fast approximation and tanh forward dispatch operations."""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad

SQRT_2_OVER_PI: float = 0.7978845608028654
COEF: float = 0.044715
