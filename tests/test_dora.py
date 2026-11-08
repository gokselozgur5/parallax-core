import numpy as np
from parallax.dora import DoRALinear


def test_dora_linear_initialization_shapes():
    layer = DoRALinear(in_features=16, out_features=32, rank=4)
    assert layer.base_V.shape == (32, 16)
    assert layer.m.shape == (32,)
    assert layer.lora_A.shape == (4, 16)
    assert layer.lora_B.shape == (32, 4)
    assert np.allclose(layer.m, np.linalg.norm(layer.base_V, axis=1))
