"""
Scalar autograd engine primitives.

Defines the foundational Value object that encapsulates a scalar value,
tracks its computation graph lineage, and stores gradient metadata.
"""

from typing import Set, Tuple, Union

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

    def __repr__(self) -> str:
        name = f"'{self.label}', " if self.label else ""
        return f"Value({name}data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __radd__(self, other: ScalarLike) -> "Value":
        return self + other

    def __mul__(self, other: ScalarLike) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __rmul__(self, other: ScalarLike) -> "Value":
        return self * other

    def __pow__(self, other: Union[int, float]) -> "Value":
        assert isinstance(other, (int, float)), "Power must be int or float"
        return Value(self.data**other, (self,), f"**{other}")

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
