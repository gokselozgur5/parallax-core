import scripts.mutation_test as mutation_test


def test_missing_mutation_target_fails_the_run(monkeypatch):
    spec = {
        "file": "parallax/optim.py",
        "target": "this text does not occur in the file",
        "mutant": "pass",
        "test": "tests/test_optim.py",
        "description": "stale target",
    }
    monkeypatch.setattr(mutation_test, "MUTATION_SPECS", [spec])
    assert mutation_test.run_mutation_test() != 0


def test_unrunnable_test_file_is_not_counted_as_a_kill(tmp_path, monkeypatch):
    (tmp_path / "pkgmod.py").write_text("X = 1\n")
    spec = {
        "file": "pkgmod.py",
        "target": "X = 1",
        "mutant": "X = 2",
        "test": "test_file_that_does_not_exist.py",
        "description": "no test to run",
    }
    monkeypatch.setattr(mutation_test, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(mutation_test, "MUTATION_SPECS", [spec])
    assert mutation_test.run_mutation_test() != 0
