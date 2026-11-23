import math

import numpy as np

from parallax.dispatch import DispatchedGELUFast, scaled_dot_product_attention
from parallax.tensor import Tensor


def test_sdpa_applies_or_rejects_attention_dropout() -> None:
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(1, 8, 16)) for _ in range(3))
    plain = scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    try:
        dropped = scaled_dot_product_attention(q, k, v, dropout_p=0.9)
    except ValueError:
        return
    assert not np.array_equal(plain, dropped), "dropout_p=0.9 changed nothing"


def test_gelu_fast_module_honours_or_rejects_exact_mode() -> None:
    x = np.array([-2.0, -1.0, 0.5, 2.0])
    exact = 0.5 * x * (1.0 + np.array([math.erf(val / math.sqrt(2.0)) for val in x]))
    try:
        layer = DispatchedGELUFast(approximate="none")
        layer(Tensor(x))
    except ValueError:
        return
    np.testing.assert_allclose(layer(Tensor(x)).data, exact, atol=1e-6)
