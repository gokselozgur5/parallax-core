from __future__ import annotations

import json
from unittest.mock import patch

from scripts import backend_info
from parallax.backends import registry


def test_backend_info_cli_default(capsys):
    registry.clear()
    ret = backend_info.main([])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "| Backend | Available | ISA Flags | Parity |" in captured
    assert "numpy" in captured

