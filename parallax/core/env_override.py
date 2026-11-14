"""Environment variable override handling for Parallax backends."""

from __future__ import annotations

import os
from collections.abc import Mapping

GLOBAL_VAR = "PARALLAX_BACKEND"
VALID_BACKENDS = frozenset({"metal", "c-neon", "c-avx", "c", "rust", "numpy", "cuda"})


def get_backend_override(
    op: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return backend override from environment if set, else None."""
    environ = os.environ if env is None else env
    raw = environ.get(GLOBAL_VAR, "").strip().lower()
    return raw if raw in VALID_BACKENDS else None
