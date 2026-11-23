"""Packaging and outside-checkout import tests for parallax.dispatch."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stage_installed_packages(tmp_path: Path) -> Path:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    find_cfg = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
    includes = find_cfg.get("include", ["parallax*"])
    installed = tmp_path / "installed"
    installed.mkdir(parents=True, exist_ok=True)
    for item in REPO_ROOT.iterdir():
        if not item.is_dir() or item.name.startswith("."):
            continue
        if any(fnmatch.fnmatch(item.name, pat) for pat in includes):
            shutil.copytree(item, installed / item.name)
    return installed


def test_parallax_dispatch_wheel_imports_outside_checkout(tmp_path: Path) -> None:
    installed = _stage_installed_packages(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(installed)
    code = (
        "import parallax.dispatch as d; "
        "missing = [n for n in d.__all__ if not hasattr(d, n)]; "
        "assert not missing, missing; "
        "assert {'DispatchedConv2d', 'DispatchedEmbeddingBackward', 'FusedSwiGLUBackward', 'FusedRMSNorm'} <= set(d.__all__)"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-400:]
    assert not (installed / "python").exists()
    assert not (REPO_ROOT / "python" / "parallax" / "dispatch").exists()
