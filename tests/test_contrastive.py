"""
Unit tests for Contrastive Search decoding primitives.
"""

import numpy as np
import pytest
from parallax.contrastive import compute_max_cosine_similarity


def test_compute_max_cosine_similarity_orthogonal_and_parallel():
    # Candidate aligned with one history vector -> similarity ~ 1.0
    history = np.array([[1.0, 0.0], [0.0, 1.0]])
    cand_same = np.array([1.0, 0.0])
    sim = compute_max_cosine_similarity(cand_same, history)
    np.testing.assert_allclose(sim, 1.0, atol=1e-6)

    # Empty history returns 0.0
    empty_history = np.empty((0, 2))
    assert compute_max_cosine_similarity(cand_same, empty_history) == 0.0
