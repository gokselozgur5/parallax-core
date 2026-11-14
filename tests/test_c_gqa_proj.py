from __future__ import annotations

import numpy as np
import pytest

from parallax.native.gqa_proj import c_gqa_proj, is_gqa_proj_available


def test_available() -> None:
    assert is_gqa_proj_available()


