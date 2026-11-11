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


def test_bitmamba_block_state_carry_over_verification():
    """Verify state carry-over is preserved; kills dropped carry-over fault."""
    np.random.seed(42)
    B, D = 1, 8
    block = BitMambaBlock(d_model=D, d_state=4, d_conv=3, expand=2)
    block.conv_weight.data = np.ones_like(block.conv_weight.data)
    block.A_log.data.fill(-0.2)
    u_step = Tensor(np.random.randn(B, 1, D))

    ssm_active = np.ones((B, block.d_inner, block.d_state)) * 2.0
    conv_active = np.zeros((B, block.d_inner, block.conv_weight.shape[1]))
    ssm_zero = np.zeros((B, block.d_inner, block.d_state))
    conv_zero = np.zeros((B, block.d_inner, block.conv_weight.shape[1]))

    with no_grad():
        out_with_carry, _, _ = block(u_step, ssm_state=ssm_active, conv_state=conv_active)
        out_without_carry, _, _ = block(u_step, ssm_state=ssm_zero, conv_state=conv_zero)

    diff = np.linalg.norm(out_with_carry.data - out_without_carry.data)
    assert diff > 0.1, "State carry-over had no effect on output"


def test_bitmamba_selective_scan_independent_recurrence_reference():
    """Verify selective scan against independent sequential loop recurrence."""
    from parallax.mamba import selective_scan_forward
    np.random.seed(42)
    B, L, D, N = 2, 4, 3, 2
    u, delta = np.random.randn(B, L, D), np.random.uniform(0.1, 0.5, (B, L, D))
    A, B_m, C_m, D_v = -np.random.uniform(0.5, 2.0, (D, N)), np.random.randn(B, L, N), np.random.randn(B, L, N), np.random.randn(D)

    y, h = selective_scan_forward(u, delta, A, B_m, C_m, D=D_v)

    h_ref = np.zeros((B, L, D, N))
    y_ref = np.zeros((B, L, D))
    h_prev = np.zeros((B, D, N))
    for i in range(L):
        dt = delta[:, i, :, None]
        h_curr = np.exp(dt * A[None, :, :]) * h_prev + (dt * B_m[:, i, None, :]) * u[:, i, :, None]
        h_ref[:, i] = h_curr
        h_prev = h_curr
        y_ref[:, i] = np.sum(C_m[:, i, None, :] * h_curr, axis=-1) + D_v[None, :] * u[:, i]

    np.testing.assert_allclose(y, y_ref, atol=1e-6)
    np.testing.assert_allclose(h, h_ref, atol=1e-6)


def test_bitmamba_docstrings_accurate():
    """Verify bitmamba module and model docstrings do not make unsupported performance claims."""
    import parallax.bitmamba as bm

    mod_doc = (bm.__doc__ or "").lower()
    lm_doc = (bm.BitMambaLM.__doc__ or "").lower()

    for claim in ["zero-ram", "sub-2-bit", "ternary additions"]:
        assert claim not in mod_doc, f"Module docstring contains unbacked claim: {claim}"
        assert claim not in lm_doc, f"BitMambaLM docstring contains unbacked claim: {claim}"


def test_bitmamba_lm_sequential_state_reference():
    """Verify BitMambaLM incremental state cache against full-sequence execution."""
    np.random.seed(42)
    config = BitMambaConfig(vocab_size=32, d_model=16, num_layers=2, d_state=4, d_conv=3, expand=2)
    model = BitMambaLM(config)

    B = 1
    idx = np.array([[3, 5, 8, 12]], dtype=np.int64)

    with no_grad():
        logits_full = model(idx).data

    states_step = [(np.zeros((B, 32, 4)), np.zeros((B, 32, 3))) for _ in range(config.num_layers)]
    step_logits = []
    with no_grad():
        for t in range(idx.shape[1]):
            token = idx[:, t : t + 1]
            logits_t, states_step = model(token, states=states_step)
            step_logits.append(logits_t.data)
    logits_step = np.concatenate(step_logits, axis=1)

    np.testing.assert_allclose(logits_full, logits_step, atol=1e-5)
    for layer_idx, (ssm, conv) in enumerate(states_step):
        assert np.any(ssm != 0.0), f"layer {layer_idx} ssm state must update"
        assert np.any(conv != 0.0), f"layer {layer_idx} conv state must update"






def test_bitmamba_multilayer_central_difference_gradient_check():
    """Verify multi-layer BitMambaLM analytical gradients against central differences (closes #435)."""
    from parallax.nn import Linear

    def make_plain_lm(config):
        lm = BitMambaLM(config)
        for block in lm.blocks:
            for name in ['in_proj', 'x_proj', 'dt_proj', 'out_proj']:
                bl = getattr(block, name)
                pl = Linear(bl.in_features, bl.out_features, bias=bl.bias is not None)
                pl.weight.data = bl.weight.data.copy()
                if pl.bias is not None:
                    pl.bias.data = bl.bias.data.copy()
                setattr(block, name, pl)
        head = lm.lm_head
        pl = Linear(head.in_features, head.out_features, bias=head.bias is not None)
        pl.weight.data = head.weight.data.copy()
        lm.lm_head = pl
        return lm

    for num_layers in [2, 3]:
        np.random.seed(42)
        cfg = BitMambaConfig(vocab_size=16, d_model=8, num_layers=num_layers, d_state=4, d_conv=3, expand=2)
        lm = make_plain_lm(cfg)
        tokens = np.array([[1, 2, 3]])
        loss = lm(tokens).sum()
        loss.backward()

        eps = 1e-6
        # Check token embedding gradient
        orig = lm.token_embeddings.weight.data[1, 0]
        lm.token_embeddings.weight.data[1, 0] = orig + eps
        l_pos = lm(tokens).sum().data
        lm.token_embeddings.weight.data[1, 0] = orig - eps
        l_neg = lm(tokens).sum().data
        lm.token_embeddings.weight.data[1, 0] = orig
        fd_emb = (l_pos - l_neg) / (2 * eps)
        an_emb = lm.token_embeddings.weight.grad[1, 0]
        np.testing.assert_allclose(an_emb, fd_emb, rtol=1e-3, atol=1e-3)

        # Check block parameters across all layers
        for b_idx, block in enumerate(lm.blocks):
            for p_name in ['in_proj', 'conv_weight', 'conv_bias', 'x_proj', 'dt_proj', 'A_log', 'D', 'out_proj']:
                param = getattr(block, p_name)
                if isinstance(param, Linear):
                    param = param.weight
                idx = (0,) * param.data.ndim
                orig = param.data[idx]
                param.data[idx] = orig + eps
                l_pos = lm(tokens).sum().data
                param.data[idx] = orig - eps
                l_neg = lm(tokens).sum().data
                param.data[idx] = orig
                fd = (l_pos - l_neg) / (2 * eps)
                an = param.grad[idx]
                np.testing.assert_allclose(an, fd, rtol=1e-3, atol=1e-3, err_msg=f'L={num_layers}, b={b_idx}, p={p_name}')
