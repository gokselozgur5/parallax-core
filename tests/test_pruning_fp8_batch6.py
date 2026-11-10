import numpy as np
import pytest

from parallax.fp8 import FP8Format, dequantize_fp8, fp8_gemm, quantize_fp8
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.prune import (
    apply_pruning_masks,
    calculate_model_sparsity,
    create_pruning_step_hook,
    prune_2_to_4_structured,
    prune_model,
    prune_unstructured_magnitude,
)
from parallax.tensor import Tensor


def test_prune_tie_breaking_deterministic():
    """Issue #286: Test magnitude pruning with tied weights reaches exact target sparsity."""
    # Case 1: All identical values
    weights_ones = np.ones((10, 10))
    pruned, mask = prune_unstructured_magnitude(weights_ones, sparsity=0.5)
    assert np.sum(pruned == 0.0) == 50
    assert np.sum(mask) == 50

    # Case 2: Ternary weights with high repetition
    np.random.seed(42)
    ternary = np.random.choice([-1.0, 0.0, 1.0], size=(20, 20))
    pruned_t, mask_t = prune_unstructured_magnitude(ternary, sparsity=0.75)
    assert np.sum(pruned_t == 0.0) == 300
    assert np.sum(mask_t) == 100

    # Case 3: Sparsity edge cases
    w = np.ones((4, 4))
    p0, m0 = prune_unstructured_magnitude(w, sparsity=0.0)
    assert np.sum(p0 == 0.0) == 0
    p1, m1 = prune_unstructured_magnitude(w, sparsity=1.0)
    assert np.sum(p1 == 0.0) == 16


def test_prune_2_to_4_sparsity_validation():
    """Issue #288: 2:4 structured pruning must validate sparsity argument."""
    weights = np.ones((4, 8))

    # sparsity=0.5 or None is accepted
    pruned, mask = prune_2_to_4_structured(weights, sparsity=0.5)
    assert np.sum(pruned == 0.0) == 16

    pruned2, mask2 = prune_2_to_4_structured(weights, sparsity=None)
    assert np.sum(pruned2 == 0.0) == 16

    # Any sparsity other than 0.5 must raise ValueError
    with pytest.raises(
        ValueError,
        match=r"2:4 structured pruning strictly requires sparsity=0\.5",
    ):
        prune_2_to_4_structured(weights, sparsity=0.6)

    with pytest.raises(
        ValueError,
        match=r"2:4 structured pruning strictly requires sparsity=0\.5",
    ):
        prune_2_to_4_structured(weights, sparsity=0.25)


def test_prune_moe_experts_and_optimizer_mask_preservation():
    """Issue #287: Prune MoE experts and maintain sparsity after optimizer step."""
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=32,
        intermediate_size=32,
        num_experts=4,
        use_swiglu=False,
    )
    model = ParallaxLM(cfg)

    # Prune only the MLP blocks (which are MoE experts)
    stats = prune_model(model, method="2:4", target_modules=["mlp"])
    # 4 experts * 2 layers (fc1, fc2) = 8 layers
    assert stats["pruned_layers_count"] == 8.0
    assert hasattr(model, "pruning_masks")
    assert len(model.pruning_masks) == 8

    # Verify each expert layer weight is pruned and has mask attribute
    for exp in model.blocks[0].mlp.experts:
        assert hasattr(exp.fc1.weight, "mask")
        assert hasattr(exp.fc2.weight, "mask")
        assert np.isclose(np.mean(exp.fc1.weight.data == 0.0), 0.5)
        assert np.isclose(np.mean(exp.fc2.weight.data == 0.0), 0.5)

    # Perform forward + backward + optimizer step
    x = Tensor(np.array([[1, 5, 10]], dtype=np.int64))
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    opt = AdamW(model.parameters(), lr=1e-2)
    opt.step()

    # Right after optimizer step, masked weights would be updated to non-zero without hook
    # Re-apply pruning masks via hook or helper
    hook = create_pruning_step_hook(model)
    hook()

    for exp in model.blocks[0].mlp.experts:
        assert np.isclose(np.mean(exp.fc1.weight.data == 0.0), 0.5)
        assert np.isclose(np.mean(exp.fc2.weight.data == 0.0), 0.5)
        assert np.all(exp.fc1.weight.data[~exp.fc1.weight.mask] == 0.0)
        assert np.all(exp.fc2.weight.data[~exp.fc2.weight.mask] == 0.0)


def test_fp8_format_type_validation():
    """Issue #267: Validate format_type normalization and error on unsupported formats."""
    x = np.random.randn(4, 16)

    # Normalization checks
    q1, s1 = quantize_fp8(x, format_type="E4M3")
    assert q1.shape == (4, 16)
    q2, s2 = quantize_fp8(x, format_type="e4m3fn")
    assert q2.shape == (4, 16)
    q3, s3 = quantize_fp8(x, format_type="E5M2")
    assert q3.shape == (4, 16)

    # Unsupported format raises ValueError
    with pytest.raises(
        ValueError,
        match=r"Unsupported FP8 format_type: 'fp16'\. Supported: 'e4m3', 'e5m2'\.",
    ):
        quantize_fp8(x, format_type="fp16")

    with pytest.raises(
        ValueError,
        match=r"Unsupported FP8 format_type: 'int8'\. Supported: 'e4m3', 'e5m2'\.",
    ):
        quantize_fp8(x, format_type="int8")


def test_fp8_mantissa_grid_emulation():
    """Issue #266: Emulate real E4M3 and E5M2 mantissa floating-point grids."""
    # Test quantizing known representable values produces zero error
    e4m3_known_values = np.array([
        0.0,
        0.001953125,  # smallest subnormal (1/512)
        0.00390625,   # subnormal 2/512
        0.015625,     # smallest normal (2^-6)
        1.0,
        1.125,        # 1 + 1/8
        1.25,         # 1 + 2/8
        448.0,        # max representable E4M3
    ])

    # If block scale is 1.0 (by having max value 448.0 in block), values in grid must be exact
    block = np.zeros(16)
    block[:len(e4m3_known_values)] = e4m3_known_values
    block[-1] = 448.0  # forces scale = 448.0 / 448.0 = 1.0

    q_block, scale = quantize_fp8(block, format_type="e4m3", block_size=16)
    deq = dequantize_fp8(q_block, scale, block.shape)
    np.testing.assert_allclose(deq[:len(e4m3_known_values)], e4m3_known_values, atol=1e-7)

    # Error on random data should match true floating point grid precision (~2-3% for E4M3, ~4-5% for E5M2)
    np.random.seed(42)
    rand_data = np.random.randn(64, 16)
    q_e4, s_e4 = quantize_fp8(rand_data, format_type="e4m3", block_size=16)
    deq_e4 = dequantize_fp8(q_e4, s_e4, rand_data.shape)
    rel_err_e4 = np.mean(np.abs(deq_e4 - rand_data) / np.abs(rand_data))
    assert 0.015 < rel_err_e4 < 0.035

    q_e5, s_e5 = quantize_fp8(rand_data, format_type="e5m2", block_size=16)
    deq_e5 = dequantize_fp8(q_e5, s_e5, rand_data.shape)
    rel_err_e5 = np.mean(np.abs(deq_e5 - rand_data) / np.abs(rand_data))
    assert 0.030 < rel_err_e5 < 0.060
