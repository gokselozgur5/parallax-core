"""PARALLAX_BACKEND environment overrides for dispatch."""

from __future__ import annotations

import os
from collections.abc import Mapping

GLOBAL_VAR = "PARALLAX_BACKEND"
VALID_BACKENDS = frozenset({"metal", "c-neon", "c-avx", "c", "rust", "numpy"})


def _normalize(var: str, raw: str) -> str | None:
    """Trim and lowercase ``raw``; empty means no override, unknown raises."""
    name = raw.strip().lower()
    if not name:
        return None
    if name not in VALID_BACKENDS:
        valid = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(f"{var}={raw!r} is not a known backend; expected one of: {valid}")
    return name
