from parallax.top_a import TopASampler


def test_top_a_sampler_init() -> None:
    sampler = TopASampler(alpha=0.3)
    assert sampler.alpha == 0.3

import numpy as np


def test_top_a_filter_logits_tail_truncation() -> None:
    sampler = TopASampler(alpha=0.5)
    # High confidence on token 0: p_max ~ 0.99
    # threshold = 0.5 * 0.99^2 ~ 0.49
    # Tokens with small probs will be discarded to -1e9
    logits = np.array([10.0, 2.0, 1.0])
    filtered = sampler.filter_logits(logits)
    assert filtered[0] == 10.0
    assert filtered[1] == -1e9
    assert filtered[2] == -1e9

def test_top_a_sample_peak() -> None:
    np.random.seed(42)
    sampler = TopASampler(alpha=0.2)
    logits = np.array([20.0, 0.0, -10.0])
    token = sampler.sample(logits)
    assert token == 0
