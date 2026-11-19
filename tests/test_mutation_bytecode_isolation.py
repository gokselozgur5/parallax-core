import subprocess
import sys
import time

import scripts.mutation_test as mutation_test


def test_mutation_run_leaves_no_mutated_bytecode(tmp_path, monkeypatch):
    (tmp_path / "pkgmod.py").write_text("def f():\n    return 2 - 1\n")
    (tmp_path / "test_pkgmod.py").write_text("import pkgmod\n\n\ndef test_f():\n    assert pkgmod.f() == 1\n")
    spec = {
        "file": "pkgmod.py",
        "target": "2 - 1",
        "mutant": "2 + 1",
        "test": "test_pkgmod.py",
        "description": "flip the operator",
    }
    monkeypatch.setattr(mutation_test, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(mutation_test, "MUTATION_SPECS", [spec])
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    probe = [sys.executable, "-c", "import pkgmod; print(pkgmod.f())"]
    for _ in range(3):
        time.sleep(1.02 - time.time() % 1.0)
        assert mutation_test.run_mutation_test() == 0
        out = subprocess.run(probe, cwd=tmp_path, capture_output=True, text=True)
        assert out.stdout.strip() == "1", "restored source still runs the mutant"
