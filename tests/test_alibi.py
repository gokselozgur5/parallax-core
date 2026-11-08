import numpy as np
from parallax.alibi import get_alibi_slopes


def test_get_alibi_slopes_shapes_and_values() -> None:
    slopes_4 = get_alibi_slopes(4)
    assert slopes_4.shape == (4,)
    assert slopes_4[0] > slopes_4[-1]
    assert np.all(slopes_4 > 0.0)
