import numpy as np
from parallax.xlora import XLoRALinear


def test_xlora_linear_initialization():
    layer = XLoRALinear(in_features=16, out_features=32, num_adapters=3, rank=2)
    assert len(layer.lora_A) == 3
    assert len(layer.lora_B) == 3
    assert layer.gate_weight.shape == (16, 3)
    assert layer.base_weight.shape == (32, 16)


def test_xlora_linear_forward_init_equals_base():
    # At initialization, lora_B is all zeros, so total_out must exactly equal base projection
    layer = XLoRALinear(in_features=8, out_features=16, num_adapters=3, rank=2)
    x = np.random.randn(8).astype(np.float32)

    out, gates = layer.forward(x)
    expected_base = np.dot(x, layer.base_weight.T)

    assert np.allclose(out, expected_base, atol=1e-6)
    assert np.isclose(np.sum(gates), 1.0)
