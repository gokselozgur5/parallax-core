import numpy as np
import pytest

from parallax.generator import generate, generate_stream
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def _create_mini_model():
    cfg = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        intermediate_size=32,
    )
    np.random.seed(42)
    return ParallaxLM(cfg)


def test_generate_stream_matches_batch_generate_deterministic():
    model = _create_mini_model()
    prompt = [1, 5, 8]
    max_tokens = 6

    # Greedy generation (temperature = 0.0)
    batch_tokens = generate(
        model,
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.0,
        use_cache=True,
    )

    stream_tokens = list(
        generate_stream(
            model,
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,
            use_cache=True,
        )
    )

    # Stream yields only newly generated tokens
    assert stream_tokens == batch_tokens[len(prompt):]
    assert len(stream_tokens) == max_tokens


def test_generate_stream_eos_early_stop():
    model = _create_mini_model()
    prompt = [1, 2]

    # Find what token would normally be generated first
    greedy_out = generate(model, prompt, max_new_tokens=4, temperature=0.0, use_cache=True)
    first_gen = greedy_out[2]

    # Set eos_token_id to that first generated token
    stream_tokens = list(
        generate_stream(
            model,
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            eos_token_id=first_gen,
            use_cache=True,
        )
    )

    assert len(stream_tokens) == 1
    assert stream_tokens[0] == first_gen

    # Also test with use_cache=False
    stream_uncached = list(
        generate_stream(
            model,
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            eos_token_id=first_gen,
            use_cache=False,
        )
    )
    assert len(stream_uncached) == 1
    assert stream_uncached[0] == first_gen


def test_generate_stream_no_cache():
    model = _create_mini_model()
    prompt = [2, 4]
    max_tokens = 4

    stream_cached = list(
        generate_stream(model, prompt, max_new_tokens=max_tokens, temperature=0.0, use_cache=True)
    )
    stream_no_cache = list(
        generate_stream(model, prompt, max_new_tokens=max_tokens, temperature=0.0, use_cache=False)
    )

    assert stream_cached == stream_no_cache
