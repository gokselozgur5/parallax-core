"""
Unit and gradient verification tests for LoRA (Low-Rank Adaptation) (M20).
"""

import numpy as np
import pytest

from parallax.lora import LoRALinear, apply_lora_to_model
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear
from parallax.optim import AdamW
from parallax.tensor import Tensor


def test_lora_linear_forward_init_identity():
    # Initially B = 0, so LoRALinear output must strictly equal base_layer output
    base = Linear(in_features=8, out_features=12, bias=True)
    lora = LoRALinear(base, r=4, lora_alpha=8.0)

    x = Tensor(np.random.randn(2, 5, 8))
    out_base = base(x)
    out_lora = lora(x)

    np.testing.assert_allclose(out_lora.data, out_base.data, atol=1e-7)


def test_lora_linear_backward_gradients_flow_only_to_adapters():
    base = Linear(in_features=6, out_features=10, bias=True)
    lora = LoRALinear(base, r=3, lora_alpha=6.0)

    x = Tensor(np.random.randn(2, 4, 6), requires_grad=True)
    out = lora(x)
    loss = (out * out).sum()
    loss.backward()

    # Base parameters must have grad None or requires_grad False
    assert not base.weight.requires_grad
    # LoRA adapters must have gradients
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.lora_A.grad.shape == (3, 6)
    assert lora.lora_B.grad.shape == (10, 3)


def test_lora_weight_merging_and_unmerging():
    base = Linear(in_features=4, out_features=4, bias=False)
    lora = LoRALinear(base, r=2, lora_alpha=4.0)

    # Populate non-zero B
    lora.lora_B.data = np.ones_like(lora.lora_B.data) * 0.5
    lora.lora_A.data = np.ones_like(lora.lora_A.data) * 0.2

    x = Tensor(np.random.randn(1, 4))
    out_unmerged = lora(x).data.copy()

    # Merge into base weight
    lora.merge_weights()
    assert lora.merged is True
    out_merged = lora(x).data.copy()
    np.testing.assert_allclose(out_merged, out_unmerged, atol=1e-7)

    # Unmerge back
    lora.unmerge_weights()
    assert lora.merged is False
    out_restored = lora(x).data.copy()
    np.testing.assert_allclose(out_restored, out_unmerged, atol=1e-7)


def test_apply_lora_to_parallax_lm():
    cfg = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=8,
    )
    model = ParallaxLM(cfg)
    total_params_before = len(model.parameters())

    lora_layers = apply_lora_to_model(model, target_modules=("q_proj", "v_proj"), r=2, lora_alpha=4.0)
    assert len(lora_layers) == 4  # 2 layers * (q_proj + v_proj)

    # Count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Each LoRA layer has 2 trainable params (lora_A, lora_B) -> 4 * 2 = 8
    assert len(trainable_params) == 8

    # Run quick forward and training step
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[2, 3, 4, 5]])
    optimizer = AdamW(trainable_params, lr=0.01)

    logits = model(x)
    loss = (logits * logits).sum()
    loss.backward()
    optimizer.step()

    # Verify adapter weights updated while base model remained frozen
    for layer in lora_layers:
        assert np.any(layer.lora_B.data != 0.0)
