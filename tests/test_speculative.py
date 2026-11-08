"""
Unit tests for Speculative Decoding from first principles (M21).
"""

import numpy as np
import pytest

from parallax.generator import generate
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.speculative import speculative_generate


def test_speculative_decoding_greedy_lossless_equivalence():
    # Set identical seed to make weights reproducible
    np.random.seed(42)
    cfg_target = ParallaxLMConfig(
        vocab_size=12,
        embed_dim=32,
        num_heads=2,
        num_layers=3,
        max_seq_len=24,
    )
    # Draft model is smaller (1 layer, embed_dim=16)
    cfg_draft = ParallaxLMConfig(
        vocab_size=12,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=24,
    )

    target_model = ParallaxLM(cfg_target)
    draft_model = ParallaxLM(cfg_draft)

    prompt = [1, 2, 3]
    max_new_tokens = 8

    # 1. Standard greedy target generation
    standard_tokens = generate(
        target_model,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        use_cache=False,
    )

    # 2. Speculative decoding with draft model
    spec_tokens, stats = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt_tokens=prompt,
        max_new_tokens=max_new_tokens,
        gamma=3,
        temperature=0.0,
    )

    # Greedy speculative decoding MUST output mathematically identical tokens as standard target model
    assert spec_tokens == standard_tokens
    assert stats["target_forward_passes"] <= max_new_tokens
    assert stats["speedup_factor"] >= 1.0


def test_speculative_decoding_identical_model_perfect_acceptance():
    # If draft model is identical to target model, acceptance rate must be 100%
    cfg = ParallaxLMConfig(
        vocab_size=10,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=20,
    )
    model = ParallaxLM(cfg)

    prompt = [2, 4]
    spec_tokens, stats = speculative_generate(
        target_model=model,
        draft_model=model,
        prompt_tokens=prompt,
        max_new_tokens=6,
        gamma=3,
        temperature=0.0,
    )
    # Perfect match on identical models
    assert stats["acceptance_rate"] == 1.0
    assert stats["speedup_factor"] > 1.0

def test_speculative_decoding_stochastic_ratio_acceptance() -> None:
    np.random.seed(42)
    cfg = ParallaxLMConfig(vocab_size=10, embed_dim=8, num_heads=2, num_layers=1, max_seq_len=16)
    target = ParallaxLM(cfg)
    draft = ParallaxLM(cfg)
    tokens, stats = speculative_generate(
        target, draft, prompt_tokens=[1, 2], max_new_tokens=4, temperature=0.8, gamma=2
    )
    assert len(tokens) >= 2
    assert "acceptance_rate" in stats
    assert 0.0 <= stats["acceptance_rate"] <= 1.0
