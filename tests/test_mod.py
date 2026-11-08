import numpy as np
from parallax.mod import MoDRouter


def test_mod_router_initialization():
    router = MoDRouter(hidden_dim=32, capacity_factor=0.5)
    assert router.hidden_dim == 32
    assert router.capacity_factor == 0.5
    assert router.weight.shape == (32, 1)
