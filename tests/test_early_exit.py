import numpy as np
from parallax.early_exit import compute_prediction_confidence


def test_compute_prediction_confidence_peak_distribution():
    # Peak distribution: token 0 dominant
    logits = np.array([20.0, 0.0, 0.0, 0.0])
    entropy, margin, top_tok = compute_prediction_confidence(logits)

    assert top_tok == 0
    assert np.isclose(entropy, 0.0, atol=1e-4)
    assert np.isclose(margin, 1.0, atol=1e-4)


from parallax.early_exit import EarlyExitClassifier


def test_early_exit_classifier_trigger():
    classifier = EarlyExitClassifier(hidden_dim=16, vocab_size=50, entropy_threshold=0.5)
    # Ensure one column of weights is massive to trigger early exit
    classifier.head_weight[:, 5] = 100.0
    hidden = np.ones(16, dtype=np.float32)

    should_exit, token, entropy = classifier.evaluate_exit(hidden)
    assert should_exit is True
    assert token == 5
    assert entropy < 0.5
