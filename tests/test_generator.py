import numpy as np
import pytest
from parallax.generator import generate_stream
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.tensor import is_grad_enabled


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


def test_generate_stream_preserves_caller_grad_state_cached():
    model = _create_mini_model()
    prompt = [1, 2, 3]

    assert is_grad_enabled() is True
    for token in generate_stream(model, prompt, max_new_tokens=3, use_cache=True):
        assert isinstance(token, int)
        assert is_grad_enabled() is True

    assert is_grad_enabled() is True


def test_generate_stream_preserves_caller_grad_state_uncached():
    model = _create_mini_model()
    prompt = [1, 2, 3]

    assert is_grad_enabled() is True
    for token in generate_stream(model, prompt, max_new_tokens=3, use_cache=False):
        assert isinstance(token, int)
        assert is_grad_enabled() is True

    assert is_grad_enabled() is True
