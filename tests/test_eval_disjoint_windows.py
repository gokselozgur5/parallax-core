import numpy as np
import pytest

from parallax.eval import compute_perplexity


class _Bigram:
    """Predicts (previous token + 1) mod 5, independent of the window."""

    def __call__(self, inputs):
        steps = inputs.shape[1]
        logits = np.zeros((1, steps, 5))
        logits[0, np.arange(steps), (inputs[0] + 1) % 5] = 8.0
        return logits


def test_disjoint_windows_score_every_target():
    tokens = [0, 1, 2, 3] * 5
    exact = compute_perplexity(_Bigram(), tokens, max_context_len=len(tokens))
    try:
        windowed = compute_perplexity(_Bigram(), tokens, max_context_len=4, stride=4)
    except ValueError:
        return  # rejecting stride >= window is an acceptable fix
    assert windowed == pytest.approx(exact, rel=1e-6)
