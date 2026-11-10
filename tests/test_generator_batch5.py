import math
import numpy as np
import pytest

from parallax.generator import generate, generate_stream, sample_next_token
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def _create_test_model(max_seq_len: int = 32):
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=max_seq_len,
        intermediate_size=32,
    )
    np.random.seed(42)
    return ParallaxLM(cfg)


def test_temperature_validation_nan_inf_subnormal():
    logits = np.array([2.0, 1.0, 0.5, 0.1])

    with pytest.raises(ValueError, match="Invalid temperature"):
        sample_next_token(logits, temperature=float("nan"))

    with pytest.raises(ValueError, match="Invalid temperature"):
        sample_next_token(logits, temperature=float("inf"))

    with pytest.raises(ValueError, match="Invalid temperature"):
        sample_next_token(logits, temperature=-0.5)

    # Subnormal temperature (e.g. 1e-320) must return greedy argmax without NaN / underflow error
    argmax_tok = int(np.argmax(logits))
    sampled_tok = sample_next_token(logits, temperature=1e-320)
    assert sampled_tok == argmax_tok


def test_top_p_threshold_mask_preserves_boundary_token():
    # probs: token 0 (0.5), token 1 (0.3), token 2 (0.2)
    # top_p = 0.7: token 0 reaches 0.5 < 0.7, token 1 reaches 0.8 >= 0.7.
    # Both token 0 and token 1 must be kept in the nucleus set.
    logits = np.log(np.array([0.5, 0.3, 0.2]))
    counts = {0: 0, 1: 0, 2: 0}

    for _ in range(200):
        tok = sample_next_token(logits, temperature=1.0, top_p=0.7)
        counts[tok] += 1

    assert counts[0] > 0
    assert counts[1] > 0
    assert counts[2] == 0


def test_cached_and_uncached_generate_reach_max_new_tokens_past_max_seq_len():
    model = _create_test_model(max_seq_len=32)
    prompt = list(range(28))

    out_uncached = generate(
        model,
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        use_cache=False,
    )
    assert len(out_uncached) == 38
    assert len(out_uncached) - len(prompt) == 10

    out_cached = generate(
        model,
        prompt,
        max_new_tokens=10,
        temperature=0.0,
        use_cache=True,
    )
    assert len(out_cached) == 38
    assert len(out_cached) - len(prompt) == 10
    assert out_cached == out_uncached

    stream_tokens = list(
        generate_stream(
            model,
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            use_cache=True,
        )
    )
    assert len(stream_tokens) == 10
    assert stream_tokens == out_cached[len(prompt):]
