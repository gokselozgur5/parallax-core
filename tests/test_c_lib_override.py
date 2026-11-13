from __future__ import annotations

from pathlib import Path
import pytest
from parallax.c_engine import (
    get_c_lib_load_error,
    is_c_engine_available,
    reset_c_lib_cache,
)


def test_c_lib_override_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reset_c_lib_cache()
    missing = tmp_path / "libnonexistent_c.dylib"
    monkeypatch.setenv("PARALLAX_C_LIB", str(missing))
    assert not is_c_engine_available()
    reason = get_c_lib_load_error()
    assert reason is not None
    assert "does not exist" in reason
    assert str(missing) in reason
    reset_c_lib_cache()


def test_c_lib_default_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_c_lib_cache()
    monkeypatch.delenv("PARALLAX_C_LIB", raising=False)
    assert isinstance(is_c_engine_available(), bool)
    reset_c_lib_cache()


def test_c_lib_override_missing_require_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reset_c_lib_cache()
    missing = tmp_path / "libmissing.dylib"
    monkeypatch.setenv("PARALLAX_C_LIB", str(missing))
    monkeypatch.setenv("PARALLAX_REQUIRE_C_ENGINE", "1")
    with pytest.raises(RuntimeError, match="PARALLAX_REQUIRE_C_ENGINE=1 but C-engine failed"):
        is_c_engine_available()
    reset_c_lib_cache()
