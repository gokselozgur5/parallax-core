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


def test_makefile_builds_and_loads_kernels():
    csrc = Path(__file__).parent.parent / "parallax" / "csrc"
    subprocess.run(["make", "-C", str(csrc), "clean"], check=True)
    try:
        subprocess.run(["make", "-C", str(csrc)], check=True)
        import numpy as np
        from parallax.c_engine import c_gemm, c_softmax, is_c_engine_available
        assert is_c_engine_available()
        a = np.ones((2, 2), dtype=np.float32)
        out = c_gemm(a, a)
        assert out is not None and np.allclose(out, [[2.0, 2.0], [2.0, 2.0]])
        sm = c_softmax(a)
        assert sm is not None and np.allclose(sm, [[0.5, 0.5], [0.5, 0.5]])
    finally:
        subprocess.run(["make", "-C", str(csrc), "clean"], check=True)

