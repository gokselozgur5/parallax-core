"""Resilient fallback execution chain across registered backends."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from parallax.core.dispatch import DispatchPolicy, PlatformInfo

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackRecord:
    """Record of a failed backend candidate and fallback reason."""

    op: str
    backend: str
    reason: str
    exception: str | None = None
