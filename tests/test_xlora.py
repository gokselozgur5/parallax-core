import numpy as np
from parallax.xlora import XLoRALinear


def test_xlora_linear_initialization():
    layer = XLoRALinear(in_features=16, out_features=32, num_adapters=3, rank=2)
    assert len(layer.lora_A) == 3
    assert len(layer.lora_B) == 3
    assert layer.gate_weight.shape == (16, 3)
    assert layer.base_weight.shape == (32, 16)
