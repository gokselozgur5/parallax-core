"""Scaled Dot-Product Attention (SDPA) dispatch operations."""

from __future__ import annotations

import math
from typing import Optional
import numpy as np

from parallax.nn import Module
from parallax.tensor import Tensor, _accumulate_grad
