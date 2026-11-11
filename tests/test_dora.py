import numpy as np
from parallax.dora import DoRALinear


def test_dora_linear_initialization_shapes():
    layer = DoRALinear(in_features=16, out_features=32, rank=4)
    assert layer.base_V.shape == (32, 16)
    assert layer.m.shape == (32,)
    assert layer.lora_A.shape == (4, 16)
    assert layer.lora_B.shape == (32, 4)
    assert np.allclose(layer.m, np.linalg.norm(layer.base_V, axis=1))


def test_dora_linear_forward_init_equals_base():
    # Because B=0 at init, W_eff = m * base_V / ||base_V|| = base_V exactly
    layer = DoRALinear(in_features=8, out_features=12, rank=2)
    x = np.random.randn(8).astype(np.float32)

    out = layer.forward(x)
    expected = np.dot(x, layer.base_V.T)

    assert np.allclose(out, expected, atol=1e-5)


def test_dora_linear_backward_shapes_and_finite():
    layer = DoRALinear(in_features=8, out_features=12, rank=2)
    x = np.random.randn(4, 8).astype(np.float32)
    grad_out = np.random.randn(4, 12).astype(np.float32)

    grad_m, grad_A, grad_B, dx = layer.backward(x, grad_out)

    assert grad_m.shape == layer.m.shape
    assert grad_A.shape == layer.lora_A.shape
    assert grad_B.shape == layer.lora_B.shape
    assert dx.shape == x.shape
    assert not np.isnan(grad_m).any()
    assert not np.isnan(grad_A).any()
    assert not np.isnan(grad_B).any()
    assert not np.isnan(dx).any()


from parallax.dora import DoRAModule
from parallax.nn import Linear
from parallax.tensor import Tensor


def test_dora_module_autograd_training():
    np.random.seed(42)
    in_features, out_features, rank = 8, 4, 2
    base = Linear(in_features, out_features, bias=True)
    dora = DoRAModule(base, rank=rank, alpha=4.0)

    x = Tensor(np.random.randn(2, in_features).astype(np.float32), requires_grad=True)
    out = dora(x)

    assert out.data.shape == (2, out_features)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, in_features)
    assert dora.m.grad is not None
    assert dora.lora_A.grad is not None
    assert dora.lora_B.grad is not None
    assert base.weight.grad is None
