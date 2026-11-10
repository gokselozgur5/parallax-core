"""
Unit tests for BitMamba: Zero-RAM Selective State Space Model with BitNet b1.58.
"""

import numpy as np
import pytest
from parallax.bitmamba import (
    BitMambaBlock,
    BitMambaConfig,
    BitMambaLM,
)
from parallax.bitnet import quantize_weights_ternary
from parallax.optim import AdamW
from parallax.tensor import Tensor, no_grad


def test_bitmamba_block_forward_shapes():
    B, T, D = 2, 8, 16
    block = BitMambaBlock(d_model=D, d_state=8, d_conv=4, expand=2)

    u = Tensor(np.random.randn(B, T, D))
    out = block(u)

    assert out.shape == (B, T, D)
    assert np.all(np.isfinite(out.data))


def test_bitmamba_block_ternary_projections():
    D = 16
    block = BitMambaBlock(d_model=D, d_state=8, d_conv=4, expand=2)

    # Verify linear projections are quantized ternary {-1, 0, 1}
    for layer in [block.in_proj, block.x_proj, block.dt_proj, block.out_proj]:
        w_ternary, gamma = quantize_weights_ternary(layer.weight.data)
        unique_vals = set(np.unique(w_ternary))
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
        assert gamma > 0.0


def test_bitmamba_block_backward_ste():
    np.random.seed(42)
    B, T, D = 2, 4, 16
    block = BitMambaBlock(d_model=D, d_state=4, d_conv=3, expand=2)

    u = Tensor(np.random.randn(B, T, D), requires_grad=True)
    out = block(u)
    loss = (out * out).sum()
    loss.backward()

    # STE gradient flow through BitLinear projections
    assert block.in_proj.weight.grad is not None
    assert block.in_proj.weight.grad.shape == block.in_proj.weight.shape
    assert np.any(block.in_proj.weight.grad != 0.0)

    assert block.x_proj.weight.grad is not None
    assert block.x_proj.weight.grad.shape == block.x_proj.weight.shape
    assert np.any(block.x_proj.weight.grad != 0.0)

    assert block.dt_proj.weight.grad is not None
    assert block.dt_proj.weight.grad.shape == block.dt_proj.weight.shape
    assert np.any(block.dt_proj.weight.grad != 0.0)

    assert block.out_proj.weight.grad is not None
    assert block.out_proj.weight.grad.shape == block.out_proj.weight.shape
    assert np.any(block.out_proj.weight.grad != 0.0)

    # Gradients to continuous conv and SSM parameters
    assert block.conv_weight.grad is not None
    assert block.conv_bias.grad is not None
    assert block.A_log.grad is not None
    assert block.D.grad is not None

    # Input gradient
    assert u.grad is not None
    assert u.grad.shape == (B, T, D)
    assert np.all(np.isfinite(u.grad))


def test_bitmamba_lm_forward_and_backward():
    np.random.seed(1337)
    config = BitMambaConfig(
        vocab_size=32,
        d_model=16,
        num_layers=2,
        d_state=4,
        d_conv=3,
        expand=2,
    )
    model = BitMambaLM(config)

    # Batch of token IDs [B=2, T=5]
    tokens = np.random.randint(0, config.vocab_size, size=(2, 5))
    logits = model(tokens)

    assert logits.shape == (2, 5, config.vocab_size)
    assert np.all(np.isfinite(logits.data))

    loss = logits.sum()
    loss.backward()

    # Embedding and LM head grad check
    assert model.token_embeddings.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    assert np.any(model.token_embeddings.weight.grad != 0.0)
    assert np.any(model.lm_head.weight.grad != 0.0)

    # First layer parameter gradients
    b0 = model.blocks[0]
    assert b0.in_proj.weight.grad is not None
    assert b0.conv_weight.grad is not None
    assert b0.A_log.grad is not None


def test_bitmamba_step_autoregressive_state_consistency():
    """Verify O(1) state-space step mode matches full sequence causal scan."""
    np.random.seed(42)
    B, T, D = 1, 4, 8
    block = BitMambaBlock(d_model=D, d_state=4, d_conv=3, expand=2)

    u_seq = np.random.randn(B, T, D)

    # Full sequence forward
    out_seq = block(Tensor(u_seq)).data

    # Step-by-step forward with recurrent states
    d_inner = block.d_inner
    ssm_state = np.zeros((B, d_inner, block.d_state))
    conv_state = np.zeros((B, d_inner, block.conv_weight.shape[1]))

    out_steps = []
    with no_grad():
        for t in range(T):
            u_t = Tensor(u_seq[:, t : t + 1, :])
            step_out, ssm_state, conv_state = block(u_t, ssm_state=ssm_state, conv_state=conv_state)
            out_steps.append(step_out.data)

    out_stepped = np.concatenate(out_steps, axis=1)

    # Verify numerical equivalence between full sequence scan and O(1) recurrent stepping
    np.testing.assert_allclose(out_seq, out_stepped, atol=1e-5, rtol=1e-5)


def test_bitmamba_step_mode_t_greater_than_one_raises():
    """Verify ValueError is raised when states are passed with T > 1."""
    block = BitMambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    ssm = np.zeros((1, block.d_inner, 4))
    conv = np.zeros((1, block.d_inner, 3))
    u = Tensor(np.random.randn(1, 5, 8))
    with no_grad():
        with pytest.raises(ValueError, match="BitMambaBlock step mode expects T=1"):
            block(u, ssm_state=ssm, conv_state=conv)


def test_bitmamba_step_mode_grad_enabled_raises():
    """Verify RuntimeError is raised when states are passed with grad enabled."""
    block = BitMambaBlock(d_model=8, d_state=4, d_conv=3, expand=2)
    ssm = np.zeros((1, block.d_inner, 4))
    conv = np.zeros((1, block.d_inner, 3))
    u = Tensor(np.random.randn(1, 1, 8))
    with pytest.raises(RuntimeError, match="BitMambaBlock step mode with state cache is inference-only"):
        block(u, ssm_state=ssm, conv_state=conv)


def test_bitmamba_default_config_logits_nonzero():
    """Verify default BitMambaLM produces non-zero logits and valid gradients."""
    np.random.seed(42)
    config = BitMambaConfig()
    model = BitMambaLM(config)
    idx = np.random.randint(0, config.vocab_size, size=(2, 5))
    logits = model(idx)
    assert np.var(logits.data) > 0.0
    loss = (logits * 0.1).sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert np.any(p.grad != 0.0)


def test_bitmamba_training_loss_reduction():
    """Verify small optimization steps strictly reduce cross-entropy loss."""
    np.random.seed(42)
    config = BitMambaConfig(vocab_size=32, d_model=16, num_layers=2, d_state=4, d_conv=3, expand=2)
    model = BitMambaLM(config)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    tokens = np.array([[1, 2, 3, 4, 5]])
    targets = np.array([[2, 3, 4, 5, 0]])

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(tokens)
        max_l = np.max(logits.data, axis=-1, keepdims=True)
        exp_l = np.exp(logits.data - max_l)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        loss_val = -np.mean(np.log(probs[0, np.arange(5), targets[0]] + 1e-12))
        losses.append(loss_val)

        grad = probs.copy()
        grad[0, np.arange(5), targets[0]] -= 1.0
        grad /= 5.0
        logits.backward(grad)
        optimizer.step()

    assert losses[-1] < losses[0]


def test_bitmamba_block_strong_ssm_regime_kill_readout_drop():
    """Verify strong SSM regime with D=0; kills SSM readout deleted fault."""
    np.random.seed(42)
    B, T, D = 1, 4, 8
    block = BitMambaBlock(d_model=D, d_state=4, d_conv=3, expand=2)
    block.D.data.fill(0.0)  # Pure SSM readout, no direct feedthrough
    block.conv_weight.data = np.ones_like(block.conv_weight.data)
    block.A_log.data.fill(-1.0)

    u_seq = np.random.randn(B, T, D) * 2.0
    out_seq = block(Tensor(u_seq)).data
    assert np.linalg.norm(out_seq) > 0.1, "Output near zero: SSM readout C.h was dropped"

    ssm_s = np.zeros((B, block.d_inner, block.d_state))
    conv_s = np.zeros((B, block.d_inner, block.conv_weight.shape[1]))
    steps = []
    with no_grad():
        for t in range(T):
            st, ssm_s, conv_s = block(Tensor(u_seq[:, t : t + 1]), ssm_state=ssm_s, conv_state=conv_s)
            steps.append(st.data)
    np.testing.assert_allclose(out_seq, np.concatenate(steps, axis=1), atol=1e-5)

