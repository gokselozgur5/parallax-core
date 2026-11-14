"""DispatchedLinear module routing matmul through backend dispatch."""

from __future__ import annotations

from typing import Any
import numpy as np

from parallax.backends.tensor_ops import dispatched_matmul
from parallax.nn import Module, Parameter
from parallax.tensor import Tensor


class DispatchedLinear(Module):
    """Drop-in nn.Linear replacement backed by dispatched_matmul."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, (out_features, in_features))
        self.weight = Parameter(weight_data, label="Linear.weight")
        self.bias: Parameter | None = Parameter(np.zeros(out_features), label="Linear.bias") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        w_t = self.weight.transpose(1, 0)
        out = dispatched_matmul(x, w_t)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_linear(cls, linear: Any) -> "DispatchedLinear":
        mod = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        mod.weight = linear.weight
        mod.bias = linear.bias
        return mod
