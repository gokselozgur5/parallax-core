import numpy as np
from parallax.mod import MoDRouter


def test_mod_router_initialization():
    router = MoDRouter(hidden_dim=32, capacity_factor=0.5)
    assert router.hidden_dim == 32
    assert router.capacity_factor == 0.5
    assert router.weight.shape == (32, 1)


def test_route_tokens_capacity_limit():
    seq_len = 10
    router = MoDRouter(hidden_dim=8, capacity_factor=0.3)  # Capacity should be ceil(3.0) = 3
    x = np.random.randn(seq_len, 8).astype(np.float32)

    indices, weights, all_scores = router.route_tokens(x)
    assert len(indices) == 3
    assert weights.shape == (3, 1)
    assert len(all_scores) == seq_len
    # Verify indices are strictly sorted
    assert np.all(np.diff(indices) > 0)
