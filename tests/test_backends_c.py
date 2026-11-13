from __future__ import annotations

from parallax.backends.base import check_backend_contract
from parallax.backends.c_backend import CBackend
from parallax.c_engine import is_c_engine_available


def test_c_backend_contract() -> None:
    backend = CBackend()
    check_backend_contract(backend)
    assert backend.name == "c"


def test_c_backend_availability() -> None:
    backend = CBackend()
    assert backend.available() == is_c_engine_available()
