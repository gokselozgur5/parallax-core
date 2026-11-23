from pathlib import Path


def test_ci_workflow_has_no_requirements_txt_reference():
    ci_yml = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    assert "requirements.txt" not in ci_yml.read_text(encoding="utf-8")


def test_ci_enforces_coverage_gate():
    ci_yml = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
    content = ci_yml.read_text(encoding="utf-8")
    assert "--cov" in content or "coverage run" in content
    assert "--cov-fail-under" in content or "--cov-report" in content

