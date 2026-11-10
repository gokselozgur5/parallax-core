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
