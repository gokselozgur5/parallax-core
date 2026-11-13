from __future__ import annotations

import sys
import numpy as np
import pytest
from parallax.backends.base import check_backend_contract
from parallax.backends.metal_backend import MetalBackend
from parallax.metal_engine import is_metal_available


def test_metal_backend_contract() -> None:
    backend = MetalBackend()
    check_backend_contract(backend)
    assert backend.name == "metal"


def test_metal_backend_availability() -> None:
    backend = MetalBackend()
    if sys.platform != "darwin":
        assert not backend.available()
    else:
        assert backend.available() == is_metal_available()
