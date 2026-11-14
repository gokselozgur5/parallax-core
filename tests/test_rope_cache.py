from __future__ import annotations
import numpy as np
import pytest
from parallax.attention import apply_rope
from parallax.native.rope_cache import RoPECache, is_rope_cache_available

def test_rope_cache_availability() -> None:
    assert is_rope_cache_available()
