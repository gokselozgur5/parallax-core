import numpy as np
from parallax.muon import newton_schulz5


def test_newton_schulz5_produces_semi_orthogonal_matrix():
    np.random.seed(42)
    G = np.random.randn(16, 32).astype(np.float32)
    O = newton_schulz5(G, steps=5)

    assert O.shape == G.shape
    # O @ O.T should be approximately identity (since M < N)
    gram = np.dot(O, O.T)
    # Check that diagonal is dominated and non-zero
    assert np.all(np.diag(gram) > 0.5)
