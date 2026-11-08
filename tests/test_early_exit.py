import numpy as np
from parallax.early_exit import compute_prediction_confidence


def test_compute_prediction_confidence_peak_distribution():
    # Peak distribution: token 0 dominant
    logits = np.array([20.0, 0.0, 0.0, 0.0])
    entropy, margin, top_tok = compute_prediction_confidence(logits)

    assert top_tok == 0
    assert np.isclose(entropy, 0.0, atol=1e-4)
    assert np.isclose(margin, 1.0, atol=1e-4)
