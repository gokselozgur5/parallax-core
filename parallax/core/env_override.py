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
    candidates = [f"{GLOBAL_VAR}_{op.upper()}"] if op else []
    candidates.append(GLOBAL_VAR)
    for var in candidates:
        raw = environ.get(var, "").strip().lower()
        if raw in VALID_BACKENDS:
            return raw
    return None
