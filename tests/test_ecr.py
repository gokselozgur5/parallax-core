from parallax.ecr import ExpertChoiceRouter


def test_ecr_initialization() -> None:
    router = ExpertChoiceRouter(num_experts=4, capacity_factor=1.5)
    assert router.num_experts == 4
    assert router.capacity_factor == 1.5
