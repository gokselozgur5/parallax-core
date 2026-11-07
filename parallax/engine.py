"""
Scalar autograd engine primitives.

Defines the foundational Value object that encapsulates a scalar value,
tracks its computation graph lineage, and stores gradient metadata.
"""

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
        assert isinstance(other, (int, float)), "Power must be int or float"
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

    def backward(self) -> None:
        """Executes backpropagation using reverse topological sort."""
        topo: List["Value"] = []
        visited: Set["Value"] = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
