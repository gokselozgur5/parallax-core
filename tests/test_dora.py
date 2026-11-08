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

    grad_m, grad_A, grad_B = layer.backward(x, grad_out)

    assert grad_m.shape == layer.m.shape
    assert grad_A.shape == layer.lora_A.shape
    assert grad_B.shape == layer.lora_B.shape
    assert not np.isnan(grad_m).any()
    assert not np.isnan(grad_A).any()
    assert not np.isnan(grad_B).any()
