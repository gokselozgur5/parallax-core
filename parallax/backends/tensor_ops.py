"""Dispatched tensor operations with autograd awareness."""

from __future__ import annotations

import numpy as np
from parallax.backends.dispatch import dispatch
from parallax.tensor import Tensor, _accumulate_grad
