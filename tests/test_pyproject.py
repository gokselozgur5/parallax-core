"""
Tests for pyproject.toml scripts, packaging metadata, and gitignore behavior.
"""

import importlib
import subprocess
import tomllib
from pathlib import Path


def test_scripts_entrypoints_importable() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    scripts = config.get("project", {}).get("scripts", {})
    assert len(scripts) == 5, f"Expected 5 scripts, found {len(scripts)}"

    for script_name, entry_str in scripts.items():
        module_name, func_name = entry_str.split(":")
        assert module_name.startswith("parallax.cli."), (
            f"Script {script_name} ({entry_str}) entrypoint must start with 'parallax.cli.'"
        )
        mod = importlib.import_module(module_name)
        func = getattr(mod, func_name)
        assert callable(func), f"Script {script_name} ({entry_str}) target is not callable"


def test_gitignore_models_unanchored_regression_829() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # probe_new_model.py inside parallax/models/ must NOT be ignored
    res_probe = subprocess.run(
        ["git", "check-ignore", "parallax/models/probe_new_model.py"],
        cwd=repo_root,
        capture_output=True,
    )
    assert res_probe.returncode == 1, "parallax/models/ must not be ignored by .gitignore (#829)"

    # root-level models/weights.bin must be ignored
    res_weights = subprocess.run(
        ["git", "check-ignore", "models/weights.bin"],
        cwd=repo_root,
        capture_output=True,
    )
    assert res_weights.returncode == 0, "root models/ directory must be ignored by .gitignore (#829)"


def test_scripts_entrypoints_no_scripts_on_sys_path() -> None:
    """Verify entrypoints import cleanly without repo-root ./scripts on sys.path."""
    import sys
    orig_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if not p.endswith("/scripts") and p != "scripts"]
        repo_root = Path(__file__).resolve().parent.parent
        with open(repo_root / "pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        for script_name, entry_str in config.get("project", {}).get("scripts", {}).items():
            module_name, func_name = entry_str.split(":")
            assert module_name.startswith("parallax.cli.")
            mod = importlib.import_module(module_name)
            func = getattr(mod, func_name)
            assert callable(func)
    finally:
        sys.path = orig_path
