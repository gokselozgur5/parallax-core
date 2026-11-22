"""Tests for parallax/csrc/Makefile AVX2 detection on Linux x86_64."""

import subprocess
import tempfile
from pathlib import Path


def _run_make_probe(fake_cpuinfo_content: str | None = None) -> tuple[str, str]:
    mk_path = Path(__file__).parent.parent / "parallax" / "csrc" / "Makefile"
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        bin_dir = tpath / "bin"
        bin_dir.mkdir()

        uname_script = bin_dir / "uname"
        uname_script.write_text(
            '#!/bin/sh\n'
            'if [ "$1" = "-m" ]; then\n'
            '    echo x86_64\n'
            'elif [ "$1" = "-s" ]; then\n'
            '    echo Linux\n'
            'else\n'
            '    echo Linux\n'
            'fi\n'
        )
        uname_script.chmod(0o755)

        fake_cpu = tpath / "cpuinfo"
        if fake_cpuinfo_content is not None:
            fake_cpu.write_text(fake_cpuinfo_content)

        grep_script = bin_dir / "grep"
        grep_script.write_text(
            '#!/bin/sh\n'
            'args=()\n'
            'for a in "$@"; do\n'
            '    if [ "$a" = "/proc/cpuinfo" ]; then\n'
            f'        args+=("{fake_cpu}")\n'
            '    else\n'
            '        args+=("$a")\n'
            '    fi\n'
            'done\n'
            'exec /usr/bin/grep "${args[@]}"\n'
        )
        grep_script.chmod(0o755)

        probe_mk = tpath / "Makefile"
        probe_mk.write_text(
            mk_path.read_text(encoding="utf-8")
            + "\nprint-probe:\n\t@echo HAS_AVX2=$(HAS_AVX2)\n\t@echo ARCH_FLAGS=$(ARCH_FLAGS)\n"
        )

        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "LC_ALL": "C"}
        res = subprocess.run(
            ["make", "-C", td, "print-probe"],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        has_avx2 = ""
        arch_flags = ""
        for line in res.stdout.splitlines():
            if line.startswith("HAS_AVX2="):
                has_avx2 = line.split("=", 1)[1].strip()
            elif line.startswith("ARCH_FLAGS="):
                arch_flags = line.split("=", 1)[1].strip()
        return has_avx2, arch_flags


def test_makefile_avx2_probe_linux_without_avx2():
    cpuinfo = "flags : fpu vme de pse tsc msr pae mce cx8 apic\n"
    has_avx2, arch_flags = _run_make_probe(cpuinfo)
    assert has_avx2 == "0", f"Expected HAS_AVX2='0', got {has_avx2!r}"
    assert "-mavx2" not in arch_flags, f"Expected no -mavx2 in ARCH_FLAGS, got {arch_flags!r}"
