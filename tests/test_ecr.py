from parallax.ecr import ExpertChoiceRouter


def test_ecr_initialization() -> None:
    router = ExpertChoiceRouter(num_experts=4, capacity_factor=1.5)
    assert router.num_experts == 4
    assert router.capacity_factor == 1.5

import numpy as np


def test_ecr_routing_and_combination() -> None:
    np.random.seed(42)
    num_tokens, num_experts, dim = 8, 2, 4
    router = ExpertChoiceRouter(num_experts=num_experts, capacity_factor=1.0)
    gating_scores = np.random.uniform(0.1, 1.0, (num_tokens, num_experts))
    indices, weights = router.route_tokens(gating_scores)

    # Capacity should be ceil(8 * 1.0 / 2) = 4
    assert indices.shape == (2, 4)
    assert weights.shape == (2, 4)

    # Dummy expert outputs of shape (num_experts, capacity, dim)
    expert_outs = np.ones((num_experts, 4, dim))
    combined = router.combine_expert_outputs(
        expert_outs, indices, weights, num_tokens=num_tokens, dim=dim
    )
    assert combined.shape == (num_tokens, dim)
    assert np.all(np.isfinite(combined))
