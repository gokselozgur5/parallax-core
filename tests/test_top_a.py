from parallax.top_a import TopASampler


def test_top_a_sampler_init() -> None:
    sampler = TopASampler(alpha=0.3)
    assert sampler.alpha == 0.3
