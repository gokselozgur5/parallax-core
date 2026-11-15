"""Tests for markdown backend availability report."""

from __future__ import annotations

from parallax.core.report import generate_markdown_report


def test_markdown_report_basic() -> None:
    report = generate_markdown_report()
    assert isinstance(report, str)
    assert "| Backend | Available | ISA Flags | Parity |" in report
    assert "|:---|:---|:---|:---|" in report


def test_markdown_report_fake_inputs() -> None:
    from parallax.backends import cpuinfo
    fake_cpu = cpuinfo.CpuInfo(arch="x86_64", system="Linux", avx2=True, avx512f=True)
    backends = {"fake_c": True, "fake_metal": False}
    parity = {"fake_c": "pass", "fake_metal": "n/a"}
    report = generate_markdown_report(
        cpu_info=fake_cpu,
        backends_override=backends,
        parity_override=parity,
    )
    expected = (
        "| Backend | Available | ISA Flags | Parity |\n"
        "|:---|:---|:---|:---|\n"
        "| fake_c | true | AVX2, AVX-512 | pass |\n"
        "| fake_metal | false | AVX2, AVX-512 | n/a |"
    )
    assert report == expected


def test_markdown_report_with_capabilities() -> None:
    backends = {"fake_c": True, "fake_rust": True}
    capabilities = {
        "fake_c": ["gemm", "softmax"],
        "fake_rust": ["attention"],
    }
    report = generate_markdown_report(
        backends=backends,
        capabilities=capabilities,
    )
    assert "| Backend | Available | ISA Flags | Parity | Capabilities |" in report
    assert "|:---|:---|:---|:---|:---|" in report
    assert "| fake_c | true |" in report
    assert "gemm, softmax" in report
    assert "| fake_rust | true |" in report
    assert "attention" in report
