"""Tests for Fused RMSNorm dispatch with epsilon stabilization."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = str(REPO_ROOT / "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from parallax.dispatch.fused_norm import stabilize_epsilon


def test_stabilize_epsilon_positive_value():
    assert stabilize_epsilon(1e-5) == 1e-5
    assert stabilize_epsilon(1e-6) == 1e-6
