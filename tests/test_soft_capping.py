from parallax.soft_capping import LogitSoftCapper


def test_logit_soft_capper_init() -> None:
    capper = LogitSoftCapper(soft_cap=50.0)
    assert capper.soft_cap == 50.0

import numpy as np


def test_soft_capper_forward_bounds() -> None:
    capper = LogitSoftCapper(soft_cap=30.0)
    extreme_logits = np.array([-1000.0, 0.0, 1000.0])
    capped = capper.forward(extreme_logits)
    assert np.isclose(capped[0], -30.0)
    assert np.isclose(capped[1], 0.0)
    assert np.isclose(capped[2], 30.0)
