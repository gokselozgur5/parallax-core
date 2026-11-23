import subprocess
from pathlib import Path


def test_gitignore_models_anchoring():
    repo_root = Path(__file__).resolve().parent.parent

    # Test 1: parallax/models/... must NOT be ignored
    res_pkg = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "parallax/models/probe_new_model.py"],
        capture_output=True,
        text=True,
    )
    assert res_pkg.returncode == 1, (
        f"parallax/models/probe_new_model.py should not be ignored: {res_pkg.stdout}"
    )

    # Test 2: root workspace models/ must be ignored
    res_root = subprocess.run(
        ["git", "-C", str(repo_root), "check-ignore", "models/weights.bin"],
        capture_output=True,
        text=True,
    )
    assert res_root.returncode == 0, (
        f"models/weights.bin should be ignored: {res_root.stderr}"
    )
