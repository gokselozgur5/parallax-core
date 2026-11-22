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


def test_collect_backend_rows_dataclass() -> None:
    from parallax.core.report import BackendReportRow, collect_backend_rows

    class DummyBackend:
        def available(self) -> bool:
            return True

        def supports(self, op: str) -> bool:
            return op in ("gemm", "rmsnorm")

    rows = collect_backend_rows({"dummy": DummyBackend()})
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, BackendReportRow)
    assert r.name == "dummy"
    assert r.available is True
    assert r.parity == "pass"
    assert r.capabilities == ("gemm", "rmsnorm")


def test_format_isa_flags_with_platform_info() -> None:
    from parallax.core.dispatch import PlatformInfo
    from parallax.core.report import format_isa_flags

    info = PlatformInfo(arch="aarch64", system="Darwin", neon=True)
    assert format_isa_flags(info) == "NEON"

    none_info = PlatformInfo(arch="unknown", system="unknown")
    assert format_isa_flags(none_info) == "none"


def test_report_does_not_claim_parity_for_a_backend_with_wrong_results() -> None:
    import numpy as np
    from parallax.backends.base import Backend
    from parallax.backends.cpuinfo import CpuInfo
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.backends.report import generate_markdown_report

    class _WrongGemm(Backend):
        name = "rust"

        def available(self) -> bool:
            return True

        def supports(self, op: str, dtype: object, shape: tuple[int, ...]) -> bool:
            return True

        def gemm(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)

    report = generate_markdown_report(
        cpu_info=CpuInfo(arch="x86_64", system="Linux"),
        backends_override={"numpy": NumpyBackend(), "rust": _WrongGemm()},
    )
    row = next(line for line in report.splitlines() if line.startswith("| rust"))
    assert "pass" not in row, row

