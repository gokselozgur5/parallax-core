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


def backend_override(op: str, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the forced backend for ``op``; the per-op variable wins."""
    env = os.environ if environ is None else environ
    per_op = f"{GLOBAL_VAR}_{op.upper()}"
    for var in (per_op, GLOBAL_VAR):
        value = _normalize(var, env.get(var, ""))
        if value is not None:
            return value
    return None
