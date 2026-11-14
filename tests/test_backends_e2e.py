import numpy as np
import pytest
from parallax.backends import registry
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.patch import disable_native, enable_native
from parallax.generator import generate
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


@pytest.fixture(autouse=True)
def setup_backend():
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()


def _create_test_model():
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


def test_native_generation_parity_greedy():
    model = _create_test_model()
    prompt = [1, 5, 12]

    np.random.seed(123)
    tokens_vanilla = generate(model, prompt, max_new_tokens=10, temperature=0.0)

    enable_native(model)
    np.random.seed(123)
    tokens_native = generate(model, prompt, max_new_tokens=10, temperature=0.0)

    assert tokens_vanilla == tokens_native
