"""
Unit tests for adapter batch 6 fixes:
- Issue #268: lora_dropout and training mode toggle
- Issue #269: apply_lora_to_model preserving existing adapters
- Issue #270: LoRALinear weight delegation, unwrap, and merge_and_unload
- Issue #274: XLoRALinear gate gradient and dL/dx in backward
- Issue #275: DoRALinear dL/dx in backward
"""

import numpy as np

from parallax.dora import DoRALinear
from parallax.lora import LoRALinear, apply_lora_to_model
from parallax.nn import Linear, Module
from parallax.tensor import Tensor
from parallax.xlora import XLoRALinear


def test_issue_268_lora_dropout_training_mode():
    np.random.seed(42)
    base = Linear(in_features=8, out_features=8, bias=False)
    layer = LoRALinear(base, r=4, lora_alpha=8.0, lora_dropout=0.5)

    # Initialize adapter weights to non-zero values to see dropout impact
    layer.lora_A.data = np.ones_like(layer.lora_A.data)
    layer.lora_B.data = np.ones_like(layer.lora_B.data)

    x = Tensor(np.ones((100, 8), dtype=np.float64))

    # In training mode, dropout is applied (outputs will vary across runs due to randomness)
    layer.training = True
    out1 = layer(x).data
    out2 = layer(x).data
    assert not np.allclose(out1, out2)

    # In eval mode (training=False), dropout must not be applied (deterministic)
    layer.training = False
    out_eval1 = layer(x).data
    out_eval2 = layer(x).data
    np.testing.assert_allclose(out_eval1, out_eval2)


def test_issue_269_preserve_existing_adapters_on_repeated_apply():
    class TwoLinearModel(Module):
        def __init__(self):
            super().__init__()
            self.q_proj = Linear(8, 8, bias=False)
            self.v_proj = Linear(8, 8, bias=False)

        def forward(self, x):
            return self.v_proj(self.q_proj(x))

    model = TwoLinearModel()

    # First apply to q_proj
    lora_layers_first = apply_lora_to_model(model, target_modules=("q_proj",), r=2)
    assert len(lora_layers_first) == 1
    assert model.q_proj.lora_A.requires_grad
    assert model.q_proj.lora_B.requires_grad

    # Second apply to v_proj; should NOT freeze model.q_proj's adapters
    lora_layers_second = apply_lora_to_model(model, target_modules=("v_proj",), r=2)
    assert len(lora_layers_second) == 1
    assert model.v_proj.lora_A.requires_grad
    assert model.v_proj.lora_B.requires_grad

    # Critical check: existing q_proj adapters must remain trainable
    assert model.q_proj.lora_A.requires_grad
    assert model.q_proj.lora_B.requires_grad


def test_issue_270_weight_delegation_unwrap_and_merge_and_unload():
    base = Linear(in_features=6, out_features=10, bias=True)
    lora = LoRALinear(base, r=3, lora_alpha=6.0)

    # Check property delegation
    assert lora.weight is base.weight
    assert lora.bias is base.bias

    # Check unwrap returns base_layer
    assert lora.unwrap() is base

    # Populate non-zero adapter weights
    lora.lora_A.data = np.full_like(lora.lora_A.data, 0.5)
    lora.lora_B.data = np.full_like(lora.lora_B.data, 0.25)
    scaling = lora.scaling

    expected_delta = (lora.lora_B.data @ lora.lora_A.data) * scaling
    original_base_weight = base.weight.data.copy()

    unloaded = lora.merge_and_unload()
    assert unloaded is base
    assert lora.merged is True
    np.testing.assert_allclose(base.weight.data, original_base_weight + expected_delta)


def test_issue_274_xlora_gate_gradient_and_dx_backward():
    np.random.seed(1337)
    layer = XLoRALinear(in_features=6, out_features=8, num_adapters=3, rank=2, alpha=4.0)
    layer.lora_B = [np.random.randn(*b.shape).astype(np.float32) * 0.1 for b in layer.lora_B]

    x = np.random.randn(4, 6).astype(np.float32)
    grad_out = np.random.randn(4, 8).astype(np.float32)

    out, gates = layer.forward(x)
    grads_A, grads_B, d_gate, dx = layer.backward(x, grad_out, gates)

    assert d_gate.shape == layer.gate_weight.shape
    assert dx.shape == x.shape
    assert not np.isnan(d_gate).any()
    assert not np.isnan(dx).any()

    # Numerical verification of gate gradient (eps=5e-3 suitable for float32 precision)
    eps = 5e-3
    d_gate_num = np.zeros_like(layer.gate_weight)
    for r in range(layer.gate_weight.shape[0]):
        for c in range(layer.gate_weight.shape[1]):
            layer.gate_weight[r, c] += eps
            y_p, _ = layer.forward(x)
            layer.gate_weight[r, c] -= 2 * eps
            y_m, _ = layer.forward(x)
            layer.gate_weight[r, c] += eps
            d_gate_num[r, c] = np.sum((y_p - y_m) * grad_out) / (2 * eps)

    np.testing.assert_allclose(d_gate, d_gate_num, rtol=1e-2, atol=1e-3)

    # Numerical verification of dx
    dx_num = np.zeros_like(x)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            x_p = x.copy()
            x_m = x.copy()
            x_p[r, c] += eps
            x_m[r, c] -= eps
            y_p, _ = layer.forward(x_p)
            y_m, _ = layer.forward(x_m)
            dx_num[r, c] = np.sum((y_p - y_m) * grad_out) / (2 * eps)

    np.testing.assert_allclose(dx, dx_num, rtol=1e-3, atol=1e-3)


def test_issue_275_dora_dx_backward():
    np.random.seed(2026)
    layer = DoRALinear(in_features=6, out_features=8, rank=3, alpha=6.0)
    layer.lora_B = np.random.randn(*layer.lora_B.shape).astype(np.float32) * 0.1

    x = np.random.randn(4, 6).astype(np.float32)
    grad_out = np.random.randn(4, 8).astype(np.float32)

    grad_m, grad_A, grad_B, dx = layer.backward(x, grad_out)

    assert dx.shape == x.shape
    assert not np.isnan(dx).any()

    # Numerical verification of dx
    eps = 1e-4
    dx_num = np.zeros_like(x)
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            x_p = x.copy()
            x_m = x.copy()
            x_p[r, c] += eps
            x_m[r, c] -= eps
            y_p = layer.forward(x_p)
            y_m = layer.forward(x_m)
            dx_num[r, c] = np.sum((y_p - y_m) * grad_out) / (2 * eps)

    np.testing.assert_allclose(dx, dx_num, rtol=1e-3, atol=1e-3)
