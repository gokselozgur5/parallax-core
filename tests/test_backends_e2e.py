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
