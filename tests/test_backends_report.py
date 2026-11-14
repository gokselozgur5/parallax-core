from __future__ import annotations

from parallax.backends.report import generate_markdown_report


def test_markdown_report_basic() -> None:
    report = generate_markdown_report()
    assert isinstance(report, str)
    assert "| Backend | Available | ISA Flags | Parity |" in report
    assert "|:---|:---|:---|:---|" in report
