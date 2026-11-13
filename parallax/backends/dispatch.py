"""Unified op dispatch with policy, environment overrides, and fallback chain."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.cpuinfo import detect_cpu
from parallax.backends.env import backend_override
from parallax.backends.policy import backend_order

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackRecord:
    """Record of a failed backend candidate and the fallback reason."""

    op: str
    backend: str
    reason: str
    exception: str | None = None
