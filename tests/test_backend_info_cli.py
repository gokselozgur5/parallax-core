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


def test_backend_info_cli_json(capsys):
    registry.clear()
    ret = backend_info.main(["--json"])
    assert ret == 0
    captured = capsys.readouterr().out
    data = json.loads(captured)
    assert "cpu" in data
    assert "backends" in data
    assert "arch" in data["cpu"]
    assert "numpy" in data["backends"]
    assert data["backends"]["numpy"]["available"] is True
    assert data["backends"]["numpy"]["parity"] == "pass"


def test_backend_info_cli_only_numpy_available(capsys):
    registry.clear()
    with patch.object(backend_info.CBackend, "available", return_value=False), \
         patch.object(backend_info.RustBackend, "available", return_value=False), \
         patch.object(backend_info.MetalBackend, "available", return_value=False):
        ret = backend_info.main([])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "| numpy | true |" in captured
        assert "| c | false |" in captured
        assert "| rust | false |" in captured
        assert "| metal | false |" in captured
