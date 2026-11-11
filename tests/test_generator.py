import numpy as np
import pytest
from parallax.generator import generate, generate_stream
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


def test_sample_next_token_temperature_scaling():
    from parallax.generator import sample_next_token

    logits = np.array([1.0, 2.0, 5.0])
    np.random.seed(42)
    samples = [sample_next_token(logits, temperature=0.05, top_k=None, top_p=None) for _ in range(50)]
    assert all(s == 2 for s in samples)


def test_generate_max_new_tokens_zero_and_negative():
    model = _create_mini_model()
    prompt = [1, 2, 3]

    assert generate(model, prompt, max_new_tokens=0, use_cache=True) == prompt
    assert generate(model, prompt, max_new_tokens=0, use_cache=False) == prompt
    assert list(generate_stream(model, prompt, max_new_tokens=0, use_cache=True)) == []
    assert list(generate_stream(model, prompt, max_new_tokens=0, use_cache=False)) == []

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1, use_cache=True)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1, use_cache=False)
    with pytest.raises(ValueError, match="max_new_tokens"):
        next(generate_stream(model, prompt, max_new_tokens=-1, use_cache=True))
    with pytest.raises(ValueError, match="max_new_tokens"):
        next(generate_stream(model, prompt, max_new_tokens=-1, use_cache=False))


def test_generate_preserves_model_training_mode():
    model = _create_mini_model()
    prompt = [1, 2, 3]

    model.train()
    assert model.training is True
    generate(model, prompt, max_new_tokens=2, use_cache=True)
    assert model.training is True

    generate(model, prompt, max_new_tokens=2, use_cache=False)
    assert model.training is True

    model.eval()
    assert model.training is False
    generate(model, prompt, max_new_tokens=2, use_cache=True)
    assert model.training is False


