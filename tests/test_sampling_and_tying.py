"""
Unit tests for Weight Tying and Nucleus (Top-P) Sampling (M12).
"""

import numpy as np
import pytest

from parallax.generator import sample_next_token
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_top_p_nucleus_sampling():
    # Construct a distribution where top 2 tokens have 90% of probability
    # logits designed to give ~ [0.60, 0.30, 0.05, 0.05]
    logits = np.array([3.0, 2.3, 0.5, 0.5])

    # Sample 50 times with top_p = 0.85
    # Only token 0 (prob ~0.62) and token 1 (prob ~0.31) should ever be chosen
    samples = set()
    for _ in range(50):
        tok = sample_next_token(logits, temperature=1.0, top_p=0.85)
        samples.add(tok)

    # Should only sample from tokens 0 and 1, never 2 or 3
    assert samples.issubset({0, 1})
    assert 0 in samples


def test_weight_tying_shares_parameters_and_reduces_count():
    # Untied model
    cfg_untied = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        tie_weights=False,
    )
    model_untied = ParallaxLM(cfg_untied)

    # Tied model
    cfg_tied = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=16,
        tie_weights=True,
    )
    model_tied = ParallaxLM(cfg_tied)

    # In tied model, token_embeddings.weight and lm_head.weight must be identical object
    assert model_tied.token_embeddings.weight is model_tied.lm_head.weight

    # Total parameters in tied model must be exactly vocab_size * embed_dim less than untied
    untied_total = sum(int(np.prod(p.shape)) for p in model_untied.parameters())
    tied_total = sum(int(np.prod(p.shape)) for p in model_tied.parameters())

    assert tied_total == untied_total - (64 * 32)


def test_weight_tying_gradient_flow():
    cfg_tied = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        tie_weights=True,
    )
    model = ParallaxLM(cfg_tied)

    idx = np.array([[1, 2, 3]])
    logits = model(idx)
    loss = logits.sum()
    loss.backward()

    # Tied weight must receive accumulated gradients from both embedding and lm_head
    weight = model.token_embeddings.weight
    assert weight.grad is not None
    assert np.any(weight.grad != 0.0)
