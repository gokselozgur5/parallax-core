import numpy as np
import pytest
from parallax.linear_attention import LinearAttention, elu_feature_map
from parallax.tensor import Tensor


def test_elu_feature_map_positivity():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 5.0])
    phi = elu_feature_map(x)
    assert np.all(phi > 0.0)
    assert np.isclose(phi[2], 1.0)
    assert np.isclose(phi[3], 2.0)


def test_linear_attention_parallel_vs_recurrent_equivalence():
    np.random.seed(42)
    B, T, C, H = 2, 8, 16, 2
    attn = LinearAttention(embed_dim=C, num_heads=H)

    x = np.random.randn(B, T, C)

    # 1. Parallel forward pass across full sequence T
    out_parallel = attn(x).data

    # 2. Recurrent single-step forward pass step-by-step
    out_recurrent_steps = []
    state = None
    for t in range(T):
        x_t = x[:, t : t + 1, :]
        out_t, state = attn(x_t, state=state, return_state=True)
        out_recurrent_steps.append(out_t.data)


    out_recurrent = np.concatenate(out_recurrent_steps, axis=1)

    assert out_parallel.shape == (B, T, C)
    assert out_recurrent.shape == (B, T, C)
    np.testing.assert_allclose(out_parallel, out_recurrent, rtol=1e-5, atol=1e-5)


def test_linear_attention_recurrent_multi_token_matches_step_by_step():
    np.random.seed(42)
    B, T, C, H = 2, 6, 16, 2
    attn = LinearAttention(embed_dim=C, num_heads=H)
    x = np.random.randn(B, T, C)

    # 1. Multi-token recurrent forward pass across full sequence T
    out_multi, final_state = attn(x, return_state=True)

    # 2. Recurrent step-by-step forward pass
    state = None
    step_outs = []
    for t in range(T):
        x_t = x[:, t : t + 1, :]
        out_t, state = attn(x_t, state=state, return_state=True)
        step_outs.append(out_t.data)
    out_step_by_step = np.concatenate(step_outs, axis=1)

    # 3. Parallel forward pass
    out_parallel = attn(x).data

    assert out_multi.shape == (B, T, C)
    np.testing.assert_allclose(out_multi.data, out_step_by_step, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_multi.data, out_parallel, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final_state[0], state[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final_state[1], state[1], rtol=1e-5, atol=1e-5)


def test_linear_attention_parallel_autograd_finite_differences():
    np.random.seed(42)
    B, T, C, H = 2, 3, 4, 2
    attn = LinearAttention(embed_dim=C, num_heads=H)
    x_data = np.random.randn(B, T, C)
    x = Tensor(x_data.copy(), requires_grad=True)

    out = attn(x)
    loss = (out * 0.5).sum()
    loss.backward()

    w_q = attn.q_proj.weight.data
    w_k = attn.k_proj.weight.data
    w_v = attn.v_proj.weight.data
    w_out = attn.out_proj.weight.data

    def run_forward(x_arr, w_q_arr, w_k_arr, w_v_arr, w_out_arr):
        D = C // H
        q = (x_arr @ w_q_arr.T).reshape(B, T, H, D).swapaxes(1, 2)
        k = (x_arr @ w_k_arr.T).reshape(B, T, H, D).swapaxes(1, 2)
        v = (x_arr @ w_v_arr.T).reshape(B, T, H, D).swapaxes(1, 2)
        phi_q = elu_feature_map(q)
        phi_k = elu_feature_map(k)
        k_col = phi_k[..., :, :, None]
        v_row = v[..., :, None, :]
        S_cum = np.cumsum(k_col @ v_row, axis=2)
        Z_cum = np.cumsum(k_col, axis=2)
        q_row = phi_q[..., :, None, :]
        num = (q_row @ S_cum).squeeze(-2)
        den = (q_row @ Z_cum).squeeze(-2) + attn.eps
        merged = (num / den).swapaxes(1, 2).reshape(B, T, C)
        return (merged @ w_out_arr.T * 0.5).sum()

    eps_fd = 1e-7
    params = [
        ("x", x_data, x.grad, lambda val: run_forward(val, w_q, w_k, w_v, w_out)),
        ("q_proj", w_q, attn.q_proj.weight.grad, lambda val: run_forward(x_data, val, w_k, w_v, w_out)),
        ("k_proj", w_k, attn.k_proj.weight.grad, lambda val: run_forward(x_data, w_q, val, w_v, w_out)),
        ("v_proj", w_v, attn.v_proj.weight.grad, lambda val: run_forward(x_data, w_q, w_k, val, w_out)),
        ("out_proj", w_out, attn.out_proj.weight.grad, lambda val: run_forward(x_data, w_q, w_k, w_v, val)),
    ]

    for name, val, auto_grad, loss_fn in params:
        assert auto_grad is not None, f"{name} grad is None"
        num_grad = np.zeros_like(val)
        it = np.nditer(val, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = val[idx]
            val[idx] = orig + eps_fd
            l_pos = loss_fn(val)
            val[idx] = orig - eps_fd
            l_neg = loss_fn(val)
            val[idx] = orig
            num_grad[idx] = (l_pos - l_neg) / (2 * eps_fd)
            it.iternext()
        abs_err = np.max(np.abs(num_grad - auto_grad))
        rel_err = np.max(np.abs(num_grad - auto_grad) / (np.abs(num_grad) + np.abs(auto_grad) + 1e-8))
        assert abs_err < 1e-5 or rel_err < 1e-4, f"Gradcheck failed for {name}: abs={abs_err}, rel={rel_err}"
