"""Unit tests for speculative decoding suite batch 6 fixes.

Covers:
- Issue #280: Generation loop bound aligns with max_seq_len.
- Issue #281: Reporting of elapsed time and accepted tokens per target step.
- Issue #282: Support for eos_token_id, top_k, top_p, and ValueError guard on gamma < 1.
- Issue #283: Root token verification against target context logits in tree drafting.
- Issue #289: Verification of architectural sketch docstrings.
"""

import numpy as np
import pytest

from parallax.early_exit import EarlyExitClassifier
from parallax.generator import generate
from parallax.medusa import MedusaModel
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.self_speculative import SelfSpeculativeEngine
from parallax.speculative import speculative_generate
from parallax.tree_drafting import verify_tree_candidates


def test_speculative_loop_bound_near_context_limit():
    np.random.seed(42)
    cfg = ParallaxLMConfig(vocab_size=20, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
    model = ParallaxLM(cfg)

    # Prompt of length 15 with max_seq_len 16
    prompt = list(range(15))
    tokens, stats = speculative_generate(
        target_model=model,
        draft_model=model,
        prompt_tokens=prompt,
        max_new_tokens=5,
        gamma=2,
        temperature=0.0,
    )
    # Target emits 1 token to reach exactly max_seq_len=16
    assert len(tokens) == 16
    assert stats["total_accepted"] == 1


def test_speculative_metrics_include_elapsed_time_and_accepted_per_step():
    np.random.seed(42)
    cfg = ParallaxLMConfig(vocab_size=15, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=20)
    model = ParallaxLM(cfg)

    tokens, stats = speculative_generate(
        target_model=model,
        draft_model=model,
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        gamma=2,
    )
    assert "elapsed_time_sec" in stats
    assert stats["elapsed_time_sec"] >= 0.0
    assert "accepted_tokens_per_step" in stats
    assert stats["accepted_tokens_per_step"] >= 0.0


def test_speculative_gamma_guard():
    cfg = ParallaxLMConfig(vocab_size=10, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
    model = ParallaxLM(cfg)

    with pytest.raises(ValueError, match="gamma must be >= 1"):
        speculative_generate(model, model, prompt_tokens=[1], gamma=0)

    with pytest.raises(ValueError, match="gamma must be >= 1"):
        speculative_generate(model, model, prompt_tokens=[1], gamma=-2)


def test_speculative_eos_token_id_stopping():
    np.random.seed(42)
    cfg = ParallaxLMConfig(vocab_size=10, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=16)
    model = ParallaxLM(cfg)

    prompt = [1, 2]
    # Predict first token to find target next token
    baseline = generate(model, prompt_tokens=prompt, max_new_tokens=1, temperature=0.0, use_cache=False)
    target_next = baseline[-1]

    tokens, stats = speculative_generate(
        target_model=model,
        draft_model=model,
        prompt_tokens=prompt,
        max_new_tokens=10,
        gamma=3,
        temperature=0.0,
        eos_token_id=target_next,
    )
    assert tokens[-1] == target_next
    assert len(tokens) == len(prompt) + 1


def test_speculative_top_k_top_p_parameters():
    np.random.seed(42)
    cfg = ParallaxLMConfig(vocab_size=20, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=20)
    model = ParallaxLM(cfg)

    tokens, stats = speculative_generate(
        target_model=model,
        draft_model=model,
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        gamma=2,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
    )
    assert len(tokens) >= 3
    assert "acceptance_rate" in stats


def test_tree_drafting_root_verification_and_correction():
    # Tree: 0 (root, token 10) -> 1 (child, token 20)
    draft_tokens = [10, 20]
    parent_indices = [-1, 0]
    target_logits = np.zeros((2, 100))
    target_logits[0, 20] = 5.0

    # Context logits predict token 99, mismatching root token 10
    context_logits = np.zeros(100)
    context_logits[99] = 10.0

    # Root token is rejected, should return correction token [99]
    res_mismatch = verify_tree_candidates(
        draft_tokens, parent_indices, target_logits, context_logits=context_logits
    )
    assert res_mismatch == [99]

    # Context logits match root token 10
    context_logits[10] = 15.0
    res_match = verify_tree_candidates(
        draft_tokens, parent_indices, target_logits, context_logits=context_logits
    )
    assert res_match == [10, 20]


def test_decoding_sketches_documentation_and_instantiation():
    import parallax.early_exit as early_exit_mod
    import parallax.medusa as medusa_mod
    import parallax.self_speculative as self_spec_mod

    assert "Architectural Verification-Loop Sketch / Prototype" in self_spec_mod.__doc__
    assert "Architectural Verification-Loop Sketch / Prototype" in medusa_mod.__doc__
    assert "Architectural Verification-Loop Sketch / Prototype" in early_exit_mod.__doc__

    engine = SelfSpeculativeEngine(exit_layer=1, total_layers=2)
    assert engine.exit_layer == 1

    medusa = MedusaModel(hidden_dim=16, vocab_size=10, num_heads=2)
    assert len(medusa.heads) == 2

    classifier = EarlyExitClassifier(hidden_dim=16, vocab_size=10)
    assert classifier.head_weight.shape == (16, 10)
