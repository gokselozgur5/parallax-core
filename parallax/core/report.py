"""Markdown backend availability report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BackendReportRow:
    """Row metadata representing backend availability and capabilities."""

    name: str
    available: bool
    isa_flags: str
    parity: str
    capabilities: tuple[str, ...] = ()
