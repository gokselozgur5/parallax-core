"""
Numerical gradient checker utility.

Uses two-sided finite differences to independently verify analytical
gradients computed by the Value autograd engine.
"""

from typing import Callable, Sequence, Union
import numpy as np
from parallax.engine import Value
from parallax.tensor import Tensor


def compute_numerical_gradient(
    fn: Callable[..., Union[float, Value]],
    inputs: Sequence[float],
    eps: float = 1e-5,
) -> list[float]:
    """Computes numerical gradient of a scalar function using central differences.

    Args:
        fn: Function taking Value inputs and returning scalar/Value.
        inputs: List of float inputs to the function.
        eps: Small perturbation for finite difference approximation.

    Returns:
        List of partial derivatives [df/dx0, df/dx1, ...].
    """
    grads = []
    for i in range(len(inputs)):
        inputs_pos = [Value(x) for x in inputs]
        inputs_neg = [Value(x) for x in inputs]

        inputs_pos[i].data += eps
        inputs_neg[i].data -= eps

        out_pos = fn(*inputs_pos)
        out_neg = fn(*inputs_neg)

        val_pos = out_pos.data if isinstance(out_pos, Value) else float(out_pos)
        val_neg = out_neg.data if isinstance(out_neg, Value) else float(out_neg)

        numerical_grad = (val_pos - val_neg) / (2.0 * eps)
        grads.append(numerical_grad)

    return grads


def check_gradient(
    fn: Callable[..., Value],
    inputs: Sequence[float],
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> bool:
    """Verifies that analytical gradients match numerical gradients within tolerance.

    Returns True if relative error for all inputs is strictly below tol.
    """
    # 1. Forward and analytical backward
    var_inputs = [Value(x) for x in inputs]
    out = fn(*var_inputs)
    out.backward()
    analytic_grads = [v.grad for v in var_inputs]

    # 2. Numerical approximation
    numerical_grads = compute_numerical_gradient(fn, inputs, eps=eps)

    # 3. Relative error comparison
    for a_grad, n_grad in zip(analytic_grads, numerical_grads):
        abs_err = abs(a_grad - n_grad)
        denom = max(abs(a_grad), abs(n_grad), 1e-8)
        rel_err = abs_err / denom
        if rel_err > tol:
            return False

    return True


def compute_numerical_tensor_gradient(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float = 1e-5
) -> list[np.ndarray]:
    """Computes numerical gradients of a scalar tensor function using central differences."""
    grads = []
    for inp in inputs:
        grad = np.zeros_like(inp.data, dtype=np.float64)
        it = np.nditer(inp.data, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = inp.data[idx]
            inp.data[idx] = orig + eps
            out_pos = float(np.sum(fn(*inputs).data))
            inp.data[idx] = orig - eps
            out_neg = float(np.sum(fn(*inputs).data))
            inp.data[idx] = orig
            grad[idx] = (out_pos - out_neg) / (2.0 * eps)
            it.iternext()
        grads.append(grad)
    return grads


def check_tensor_gradient(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], eps: float = 1e-5, tol: float = 1e-4
) -> bool:
    """Verifies Tensor analytical gradients against finite differences."""
    for inp in inputs:
        inp.grad = None
    out = fn(*inputs)
    out.backward()
    analytic_grads = [inp.grad.copy() if inp.grad is not None else np.zeros_like(inp.data) for inp in inputs]
    numerical_grads = compute_numerical_tensor_gradient(fn, inputs, eps=eps)
    for a_grad, n_grad in zip(analytic_grads, numerical_grads):
        denom = np.maximum(np.maximum(np.abs(a_grad), np.abs(n_grad)), 1e-7)
        if np.max(np.abs(a_grad - n_grad) / denom) > tol:
            return False
    return True
