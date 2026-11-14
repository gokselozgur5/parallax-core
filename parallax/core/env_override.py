"""Environment variable override handling for Parallax backends."""

from __future__ import annotations

import os
from collections.abc import Mapping

GLOBAL_VAR = "PARALLAX_BACKEND"
VALID_BACKENDS = frozenset({"metal", "c-neon", "c-avx", "c", "rust", "numpy", "cuda"})


def _normalize(var: str, raw: str) -> str | None:
    name = raw.strip().lower()
    if not name:
        return None
    if name not in VALID_BACKENDS:
        valid = ", ".join(sorted(VALID_BACKENDS))
        raise ValueError(f"{var}={raw!r} is not a known backend; expected one of: {valid}")
    return name


def get_backend_override(
    op: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return backend override from environment if set, else None."""
    environ = os.environ if env is None else env
    candidates = [f"{GLOBAL_VAR}_{op.upper()}"] if op else []
    candidates.append(GLOBAL_VAR)
    for var in candidates:
        if var in environ:
            val = _normalize(var, environ[var])
            if val is not None:
                return val
    return None


backend_override = get_backend_override


def is_override_active(
    op: str | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return True if an environment override is specified for the backend."""
    return get_backend_override(op=op, env=env) is not None
