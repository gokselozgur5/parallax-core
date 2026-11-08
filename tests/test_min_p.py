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
