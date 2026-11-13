"""Tests for parallax/csrc/Makefile wildcard sources and ISA flags."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def test_makefile_wildcard_and_isa_rules():
    mk_path = Path(__file__).parent.parent / "parallax" / "csrc" / "Makefile"
    content = mk_path.read_text(encoding="utf-8")
    assert "wildcard *.c" in content
    assert "AVX512_FLAGS" in content
    assert "%_avx512.o" in content
    assert "CFLAGS_ISA" in content


def test_makefile_avx512_flag_injection():
    mk = Path(__file__).parent.parent / "parallax" / "csrc" / "Makefile"
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        shutil.copy(mk, tpath / "Makefile")
        (tpath / "test_avx512.c").touch()
        cmd = ["make", "-C", td, "-n", "test_avx512.o", "UNAME_M=x86_64"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "-mavx512f" in res.stdout
