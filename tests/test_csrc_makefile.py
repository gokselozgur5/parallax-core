"""Tests for parallax/csrc/Makefile wildcard sources and ISA flags."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_in_temp_csrc(code: str, run_clean: bool = False) -> None:
    root = Path(__file__).parent.parent
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        csrc = t / "parallax" / "csrc"
        shutil.copytree(root / "parallax" / "csrc", csrc)
        (t / "parallax" / "__init__.py").write_text("")
        shutil.copy2(root / "parallax" / "c_engine.py", t / "parallax" / "c_engine.py")
        env = dict(os.environ)
        if sys.platform == "darwin" and "SDKROOT" not in env:
            for s in ("MacOSX26.5.sdk", "MacOSX12.3.sdk"):
                p = Path(f"/Library/Developer/CommandLineTools/SDKs/{s}")
                if p.exists():
                    env["SDKROOT"] = str(p)
                    break
        if run_clean:
            subprocess.run(["make", "-C", str(csrc), "clean"], env=env, check=True)
        try:
            subprocess.run([sys.executable, "-c", code], cwd=str(t), env=env, check=True)
        finally:
            if run_clean:
                subprocess.run(["make", "-C", str(csrc), "clean"], env=env, check=True)


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
    code = (
        "import subprocess, numpy as np, parallax.c_engine as ce\n"
        "from pathlib import Path\n"
        "p = Path(ce.__file__).parent / 'csrc'\n"
        "subprocess.run(['make', '-C', str(p)], check=True)\n"
        "assert ce.is_c_engine_available()\n"
        "a = np.ones((2, 2), dtype=np.float32)\n"
        "out = ce.c_gemm(a, a)\n"
        "assert out is not None and np.allclose(out, [[2.0, 2.0], [2.0, 2.0]])\n"
        "sm = ce.c_softmax(a)\n"
        "assert sm is not None and np.allclose(sm, [[0.5, 0.5], [0.5, 0.5]])\n"
    )
    _run_in_temp_csrc(code, run_clean=True)
