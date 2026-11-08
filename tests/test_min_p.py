import numpy as np
from parallax.min_p import min_p_filter


def test_min_p_filter_relative_threshold():
    logits = np.array([10.0, 5.0, 1.0, 0.0])  # Dominant token at 0
    probs = min_p_filter(logits, min_p=0.1, temperature=1.0)

    # Token 0 is dominant; tokens with tiny prob should be truncated
    assert probs[0] > 0.9
    assert probs[2] == 0.0
    assert probs[3] == 0.0
    assert np.isclose(np.sum(probs), 1.0)


from parallax.min_p import sample_min_p


def test_sample_min_p_determinism_with_rng():
    logits = np.array([2.0, 1.0, 0.5, -1.0])
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)

    tok1 = sample_min_p(logits, min_p=0.2, rng=rng1)
    tok2 = sample_min_p(logits, min_p=0.2, rng=rng2)

    assert tok1 == tok2
    assert 0 <= tok1 < 4
