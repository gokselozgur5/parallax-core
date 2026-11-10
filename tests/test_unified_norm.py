import numpy as np
import pytest
from parallax.unified_norm import UnifiedNorm


def test_unified_norm_initialization() -> None:
    norm_rms = UnifiedNorm(dim=32, norm_type="rmsnorm")
    assert norm_rms.dim == 32
    assert norm_rms.bias is None

    norm_ln = UnifiedNorm(dim=32, norm_type="layernorm")
    assert norm_ln.bias is not None
    assert norm_ln.bias.shape == (32,)


def test_unified_norm_fused_residual_rmsnorm() -> None:
    norm = UnifiedNorm(dim=4, norm_type="rmsnorm")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    res = np.array([[1.0, 1.0, 1.0, 1.0]])
    normed, fused = norm.forward(x, residual=res)
    assert np.allclose(fused, [[2.0, 3.0, 4.0, 5.0]])
    assert normed.shape == (1, 4)
    assert np.all(np.isfinite(normed))


def test_unified_norm_backward_finite_and_shapes() -> None:
    norm = UnifiedNorm(dim=4, norm_type="layernorm")
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    normed, fused = norm.forward(x)
    grad_out = np.ones_like(normed)
    grad_x, grad_w, grad_b = norm.backward(grad_out, fused)
    assert grad_x.shape == x.shape
    assert grad_w.shape == (4,)
    assert grad_b is not None and grad_b.shape == (4,)
    assert np.all(np.isfinite(grad_x))


@pytest.mark.parametrize("norm_type", ["rmsnorm", "layernorm"])
def test_unified_norm_backward_finite_differences(norm_type: str) -> None:
    np.random.seed(42)
    dim = 4
    x = np.random.randn(2, dim)
    norm = UnifiedNorm(dim=dim, norm_type=norm_type)
    norm.weight = np.random.randn(dim)
    if norm_type == "layernorm":
        norm.bias = np.random.randn(dim)

    normed, fused = norm.forward(x)
    grad_out = np.random.randn(*normed.shape)
    grad_x, grad_w, grad_b = norm.backward(grad_out, fused)

    eps = 1e-6
    # Numerical grad wrt x
    num_grad_x = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_pos = x.copy()
            x_pos[i, j] += eps
            n_pos, _ = norm.forward(x_pos)
            x_neg = x.copy()
            x_neg[i, j] -= eps
            n_neg, _ = norm.forward(x_neg)
            num_grad_x[i, j] = np.sum(grad_out * (n_pos - n_neg)) / (2.0 * eps)
    np.testing.assert_allclose(grad_x, num_grad_x, rtol=1e-4, atol=1e-4)

    # Numerical grad wrt weight
    num_grad_w = np.zeros_like(norm.weight)
    for j in range(dim):
        orig = norm.weight[j]
        norm.weight[j] = orig + eps
        n_pos, _ = norm.forward(x)
        norm.weight[j] = orig - eps
        n_neg, _ = norm.forward(x)
        norm.weight[j] = orig
        num_grad_w[j] = np.sum(grad_out * (n_pos - n_neg)) / (2.0 * eps)
    np.testing.assert_allclose(grad_w, num_grad_w, rtol=1e-4, atol=1e-4)

    if norm_type == "layernorm":
        assert grad_b is not None
        num_grad_b = np.zeros_like(norm.bias)
        for j in range(dim):
            orig = norm.bias[j]
            norm.bias[j] = orig + eps
            n_pos, _ = norm.forward(x)
            norm.bias[j] = orig - eps
            n_neg, _ = norm.forward(x)
            norm.bias[j] = orig
            num_grad_b[j] = np.sum(grad_out * (n_pos - n_neg)) / (2.0 * eps)
        np.testing.assert_allclose(grad_b, num_grad_b, rtol=1e-4, atol=1e-4)
    else:
        assert grad_b is None
