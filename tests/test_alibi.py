import numpy as np
import pytest
from parallax.alibi import alibi_causal_attention, build_alibi_bias_matrix, get_alibi_slopes


def test_get_alibi_slopes_shapes_and_values() -> None:
    slopes_4 = get_alibi_slopes(4)
    assert slopes_4.shape == (4,)
    assert slopes_4[0] > slopes_4[-1]
    assert np.all(slopes_4 > 0.0)


@pytest.mark.parametrize("num_heads", [3, 5, 6, 10, 12])
def test_get_alibi_slopes_non_power_of_2(num_heads: int) -> None:
    slopes = get_alibi_slopes(num_heads)
    assert slopes.shape == (num_heads,)
    assert np.all(slopes > 0.0)
    assert np.all(np.isfinite(slopes))

    closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
    # First `closest_power_of_2` slopes match standard power-of-2 slopes
    base_slopes = get_alibi_slopes(closest_power_of_2)
    np.testing.assert_allclose(slopes[:closest_power_of_2], base_slopes)


def test_build_alibi_bias_matrix_diagonal() -> None:
    bias = build_alibi_bias_matrix(num_heads=2, seq_len=4)
    assert bias.shape == (2, 4, 4)
    # Diagonal should be 0 since relative distance is 0
    for h in range(2):
        assert np.all(np.diag(bias[h]) == 0.0)


def test_alibi_causal_attention_output() -> None:
    np.random.seed(42)
    num_heads, seq_len, head_dim = 2, 8, 16
    q = np.random.randn(num_heads, seq_len, head_dim)
    k = np.random.randn(num_heads, seq_len, head_dim)
    v = np.random.randn(num_heads, seq_len, head_dim)
    out = alibi_causal_attention(q, k, v)
    assert out.shape == (num_heads, seq_len, head_dim)
    assert np.all(np.isfinite(out))


def test_alibi_causal_attention_non_power_of_2() -> None:
    np.random.seed(42)
    num_heads, seq_len, head_dim = 6, 8, 16
    q = np.random.randn(num_heads, seq_len, head_dim)
    k = np.random.randn(num_heads, seq_len, head_dim)
    v = np.random.randn(num_heads, seq_len, head_dim)
    out = alibi_causal_attention(q, k, v)
    assert out.shape == (num_heads, seq_len, head_dim)
    assert np.all(np.isfinite(out))
