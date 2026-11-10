"""
Scalar autograd engine primitives.

Defines the foundational Value object that encapsulates a scalar value,
tracks its computation graph lineage, and stores gradient metadata.
"""

import math
from typing import Callable, List, Set, Tuple, Union

ScalarLike = Union[int, float, "Value"]


class Value:
    """Scalar node in a dynamic computation graph."""

    def __init__(
        self,
        data: float,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: Set["Value"] = set(_children)
        self._op: str = _op
        self.label: str = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        name = f"'{self.label}', " if self.label else ""
        return f"Value({name}data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: ScalarLike) -> "Value":
        return self + other

    def __mul__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: ScalarLike) -> "Value":
        return self * other

    def __pow__(self, other: Union[int, float]) -> "Value":
        if not isinstance(other, (int, float)):
            raise TypeError("Power must be int or float")
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: ScalarLike) -> "Value":
        return self + (-other)

    def __rsub__(self, other: ScalarLike) -> "Value":
        return (-self) + other

    def __truediv__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * (other**-1.0)

    def __rtruediv__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def exp(self) -> "Value":
        clamped_data = max(-700.0, min(self.data, 700.0))
        out = Value(math.exp(clamped_data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError("Logarithm undefined for non-positive values")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        if self.data >= 0:
            s = 1.0 / (1.0 + math.exp(-self.data))
        else:
            exp_x = math.exp(self.data)
            s = exp_x / (1.0 + exp_x)
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def gelu(self) -> "Value":
        """Gaussian Error Linear Unit (standard tanh approximation)."""
        x = self.data
        cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))
        out = Value(x * cdf, (self,), "GELU")

        def _backward() -> None:
            # Analytical derivative of GELU approximation
            inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))
            tanh_val = math.tanh(inner)
            sech2_val = 1.0 - tanh_val**2
            d_inner = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * (x**2))
            d_cdf = 0.5 * sech2_val * d_inner
            self.grad += (cdf + x * d_cdf) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Executes backpropagation using reverse topological sort."""
        topo: List["Value"] = []
        visited: Set["Value"] = set()
        stack: List[Tuple["Value", bool]] = [(self, False)]

        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                if child not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
