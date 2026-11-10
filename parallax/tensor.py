"""
N-Dimensional Tensor autograd engine.

Wraps NumPy ndarrays into dynamic computation graph nodes, providing
automatic differentiation with broadcast-aware gradient reduction,
matrix multiplication, and common reduction operators.
"""

import contextlib
import contextvars
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

TensorLike = Union[int, float, list, np.ndarray, "Tensor"]

_grad_enabled_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "grad_enabled", default=True
)


def is_grad_enabled() -> bool:
    """Returns True if autograd computation is enabled in the current context."""
    return _grad_enabled_var.get()


@contextlib.contextmanager
def no_grad():
    """Context manager that disables gradient calculation in current context."""
    token = _grad_enabled_var.set(False)
    try:
        yield
    finally:
        _grad_enabled_var.reset(token)


def unbroadcast_grad(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Reduces a gradient array to match target shape after broadcasting."""
    grad_shape = grad.shape
    if grad_shape == target_shape:
        return grad

    # Sum along leading dimensions if target has fewer dimensions
    grad_ndim = len(grad_shape)
    target_ndim = len(target_shape)
    diff = grad_ndim - target_ndim
    if diff > 0:
        grad = grad.sum(axis=tuple(range(diff)))

    # Sum along dimensions that were broadcasted from 1 to N
    for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


def _accumulate_grad(tensor: "Tensor", grad: np.ndarray) -> None:
    """Accumulate gradient into tensor lazily."""
    if not tensor.requires_grad:
        return
    g = unbroadcast_grad(grad, tensor.shape)
    if tensor.grad is None:
        tensor.grad = np.array(g, dtype=np.float64)
    else:
        tensor.grad += g


class Tensor:
    """N-dimensional tensor supporting reverse-mode automatic differentiation."""

    __array_ufunc__ = None

    def __init__(
        self,
        data: TensorLike,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        requires_grad: bool = True,
        label: str = "",
    ):
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data.astype(np.float64, copy=False)
        else:
            self.data = np.array(data, dtype=np.float64)

        grad_enabled = _grad_enabled_var.get()
        if _children:
            self.requires_grad: bool = grad_enabled and any(c.requires_grad for c in _children)
        else:
            self.requires_grad: bool = bool(requires_grad) and grad_enabled

        self.grad: Optional[np.ndarray] = None
        self._prev: Set["Tensor"] = set(_children) if self.requires_grad else set()
        self._op: str = _op
        self.label: str = label
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        name = f"'{self.label}', " if self.label else ""
        return f"Tensor({name}shape={self.shape}, op='{self._op}')"

    def zero_grad(self) -> None:
        self.grad = None

    def __add__(self, other: TensorLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            if out.grad is None:
                return
            _accumulate_grad(self, out.grad)
            _accumulate_grad(other, out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other: TensorLike) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: TensorLike) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: TensorLike) -> "Tensor":
        return (-self) + other

    def __mul__(self, other: TensorLike) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                _accumulate_grad(self, out.grad * other.data)
            if other.requires_grad:
                _accumulate_grad(other, out.grad * self.data)

        out._backward = _backward
        return out

    def __rmul__(self, other: TensorLike) -> "Tensor":
        return self * other

    def __matmul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            raise TypeError("Matmul requires another Tensor")
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            if out.grad is None:
                return
            a_is_1d, b_is_1d = (self.ndim == 1), (other.ndim == 1)
            a_2d = self.data[None, :] if a_is_1d else self.data
            b_2d = other.data[..., :, None] if b_is_1d else other.data

            if a_is_1d and b_is_1d:
                g_2d = np.array([[float(out.grad)]])
            elif a_is_1d:
                g_2d = out.grad[..., None, :]
            elif b_is_1d:
                g_2d = out.grad[..., None]
            else:
                g_2d = out.grad

            if self.requires_grad:
                other_t = np.swapaxes(b_2d, -1, -2)
                grad_self = g_2d @ other_t
                if a_is_1d:
                    grad_self = np.squeeze(grad_self, axis=-2)
                _accumulate_grad(self, grad_self)

            if other.requires_grad:
                self_t = np.swapaxes(a_2d, -1, -2)
                grad_other = self_t @ g_2d
                if b_is_1d:
                    grad_other = np.squeeze(grad_other, axis=-1)
                _accumulate_grad(other, grad_other)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        if not axes:
            # Default swap last two dims
            axes = tuple(range(self.ndim - 2)) + (self.ndim - 1, self.ndim - 2) if self.ndim >= 2 else (0,)
        norm_axes = tuple(ax % self.ndim for ax in axes)
        out = Tensor(np.transpose(self.data, norm_axes), (self,), "transpose")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                inv_axes = np.argsort(norm_axes)
                _accumulate_grad(self, np.transpose(out.grad, inv_axes))

        out._backward = _backward
        return out

    def sum(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> "Tensor":
        if axis is not None:
            if isinstance(axis, (int, np.integer)):
                norm_axis = int(axis)
            else:
                norm_axis = tuple(int(a) for a in axis)
        else:
            norm_axis = None
        out_data = self.data.sum(axis=norm_axis, keepdims=keepdims)
        out = Tensor(out_data, (self,), "sum")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad = out.grad
                if not keepdims and norm_axis is not None:
                    # Expand dimensions back for broadcasting
                    axes = (norm_axis,) if isinstance(norm_axis, int) else norm_axis
                    shape = list(self.shape)
                    for ax in axes:
                        shape[ax] = 1
                    grad = grad.reshape(shape)
                _accumulate_grad(self, np.broadcast_to(grad, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Sequence[int]]] = None, keepdims: bool = False) -> "Tensor":
        if axis is not None:
            if isinstance(axis, (int, np.integer)):
                norm_axis = int(axis)
            else:
                norm_axis = tuple(int(a) for a in axis)
        else:
            norm_axis = None
        s = self.sum(axis=norm_axis, keepdims=keepdims)
        num_elements = self.data.size if norm_axis is None else (
            np.prod([self.shape[ax] for ax in ((norm_axis,) if isinstance(norm_axis, int) else norm_axis)])
        )
        return s * (1.0 / num_elements)

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                _accumulate_grad(self, (self.data > 0.0) * out.grad)

        out._backward = _backward
        return out

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """Executes backpropagation through the tensor DAG."""
        if grad is None:
            if not (self.shape == () or self.shape == (1,)):
                raise ValueError("Default backward only supported for scalar outputs")
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = grad.astype(np.float64)

        topo: List["Tensor"] = []
        visited: Set["Tensor"] = set()
        stack: List[Tuple["Tensor", bool]] = [(self, False)]

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

        for node in reversed(topo):
            node._backward()
