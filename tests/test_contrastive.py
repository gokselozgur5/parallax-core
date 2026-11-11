"""
Unit tests for Contrastive Search decoding primitives.
"""

import numpy as np
from parallax.contrastive import (
    compute_max_cosine_similarity,
    generate_contrastive,
    score_contrastive_candidate,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_compute_max_cosine_similarity_orthogonal_and_parallel():
    # Candidate aligned with one history vector -> similarity ~ 1.0
    history = np.array([[1.0, 0.0], [0.0, 1.0]])
    cand_same = np.array([1.0, 0.0])
    sim = compute_max_cosine_similarity(cand_same, history)
    np.testing.assert_allclose(sim, 1.0, atol=1e-6)

    # Empty history returns 0.0
    empty_history = np.empty((0, 2))
    assert compute_max_cosine_similarity(cand_same, empty_history) == 0.0


def test_score_contrastive_candidate_penalizes_repetition():
    history = np.array([[1.0, 0.0], [0.8, 0.2]])
    # Candidate 1: High probability (0.9), but identical to history
    score_repetitive = score_contrastive_candidate(
        token_prob=0.9, candidate_vec=np.array([1.0, 0.0]), history_matrix=history, alpha=0.6
    )
    # Candidate 2: Moderate probability (0.6), but orthogonal/novel to history
    score_novel = score_contrastive_candidate(
        token_prob=0.6, candidate_vec=np.array([0.0, 1.0]), history_matrix=history, alpha=0.6
    )
    # The novel token must receive a higher overall score than the repetitive token
    assert score_novel > score_repetitive


def test_generate_contrastive_end_to_end():
    np.random.seed(42)
    cfg = ParallaxLMConfig(
        vocab_size=12,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=20,
    )
    model = ParallaxLM(cfg)

    prompt = [1, 2, 3]
    generated = generate_contrastive(model, prompt, max_new_tokens=5, k=3, alpha=0.6)

    assert len(generated) == len(prompt) + 5
    assert generated[: len(prompt)] == prompt
    assert all(0 <= t < cfg.vocab_size for t in generated)


