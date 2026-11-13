from __future__ import annotations

from pathlib import Path
import pytest
from parallax.rust_engine import find_rust_lib


def test_rust_lib_override_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_lib = tmp_path / "libcustom_rs.dylib"
    fake_lib.write_text("dummy")
    monkeypatch.setenv("PARALLAX_RUST_LIB", str(fake_lib))
    found = find_rust_lib()
    assert found == fake_lib.resolve()


def test_rust_lib_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARALLAX_RUST_LIB", raising=False)
    found = find_rust_lib()
    if found is not None:
        assert found.is_file()
