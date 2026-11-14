from __future__ import annotations

from parallax.backends import cpuinfo
from parallax.backends.report import generate_markdown_report


def test_markdown_report_basic() -> None:
    report = generate_markdown_report()
    assert isinstance(report, str)
    assert "| Backend | Available | ISA Flags | Parity |" in report
    assert "|:---|:---|:---|:---|" in report


def test_markdown_report_fake_inputs() -> None:
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
