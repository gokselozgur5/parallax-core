import numpy as np
from parallax.alibi import get_alibi_slopes


def test_get_alibi_slopes_shapes_and_values() -> None:
    slopes_4 = get_alibi_slopes(4)
    assert slopes_4.shape == (4,)
    assert slopes_4[0] > slopes_4[-1]
    assert np.all(slopes_4 > 0.0)

from parallax.alibi import build_alibi_bias_matrix


def test_build_alibi_bias_matrix_diagonal() -> None:
    bias = build_alibi_bias_matrix(num_heads=2, seq_len=4)
    assert bias.shape == (2, 4, 4)
    # Diagonal should be 0 since relative distance is 0
    for h in range(2):
        assert np.all(np.diag(bias[h]) == 0.0)

from parallax.alibi import alibi_causal_attention


def test_alibi_causal_attention_output() -> None:
    np.random.seed(42)
    num_heads, seq_len, head_dim = 2, 8, 16
    q = np.random.randn(num_heads, seq_len, head_dim)
    k = np.random.randn(num_heads, seq_len, head_dim)
    v = np.random.randn(num_heads, seq_len, head_dim)
    out = alibi_causal_attention(q, k, v)
    assert out.shape == (num_heads, seq_len, head_dim)
    assert np.all(np.isfinite(out))
