"""Dispatched LayerNorm with RMSNorm weight scaling and bias fusion."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad
