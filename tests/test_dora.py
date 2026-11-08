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
