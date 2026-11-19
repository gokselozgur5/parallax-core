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


from parallax.mod import mod_block_forward


def test_mod_block_forward_identity_bypass_for_unselected():
    seq_len, hidden_dim = 6, 4
    x = np.random.randn(seq_len, hidden_dim).astype(np.float32)
    router = MoDRouter(hidden_dim=hidden_dim, capacity_factor=0.5)

    # Dummy block adding 10.0 to selected tokens
    def dummy_block(h):
        return h + 10.0

    output, selected_indices = mod_block_forward(x, router, dummy_block)

    assert output.shape == x.shape
    # Check that unselected tokens remain strictly unchanged (identity bypass)
    unselected = [i for i in range(seq_len) if i not in selected_indices]
    for idx in unselected:
        assert np.allclose(output[idx], x[idx])


def test_causal_routing_output_at_t_ignores_later_tokens():
    rng = np.random.default_rng(0)
    router = MoDRouter(hidden_dim=4, threshold=0.99)
    router.weight = np.full((4, 1), 0.1)
    x_low = rng.standard_normal((6, 4))
    x_high = x_low.copy()
    x_low[3:] -= 5.0
    x_high[3:] += 5.0

    def block(h: np.ndarray) -> np.ndarray:
        return h + 10.0

    out_low, picked_low = mod_block_forward(x_low, router, block, causal=True)
    out_high, picked_high = mod_block_forward(x_high, router, block, causal=True)
    assert np.all(router.route_tokens(x_high, causal=True)[2] < 0.99)
    np.testing.assert_allclose(
        out_low[:3], out_high[:3], err_msg=f"{picked_low} vs {picked_high}"
    )
