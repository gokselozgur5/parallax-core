from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from parallax.backends.cpuinfo import detect_cpu
from parallax.native import avx2_rmsnorm
from parallax.native.avx2_rmsnorm import is_avx2_rmsnorm_available
from parallax.native.avx2_softmax import is_avx2_softmax_available, softmax_avx2
from parallax.native.gemm_avx2 import gemm_avx2, is_avx2_gemm_available
from parallax.native.loader import bind


def test_avx2_availability_and_status_codes() -> None:
    fp, ci, cf = ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float
    c_rms = bind("c_rmsnorm_avx2_f32", [fp, fp, fp, ci, ci, cf], ci)
    c_gemm = bind("c_gemm_avx2_f32", [fp, fp, fp, ci, ci, ci], ci)
    c_smax = bind("c_softmax_avx2_f32", [fp, fp, ci, ci], ci)
    assert c_rms is not None and c_gemm is not None and c_smax is not None
    buf, out = np.ones((2, 8), dtype=np.float32), np.empty((2, 8), dtype=np.float32)
    ptr, out_p, null = buf.ctypes.data_as(fp), out.ctypes.data_as(fp), ctypes.cast(0, fp)
    if not detect_cpu().avx2:
        assert not is_avx2_rmsnorm_available()
    supported = is_avx2_rmsnorm_available()
    assert is_avx2_gemm_available() is supported and is_avx2_softmax_available() is supported
    assert (c_rms(null, ptr, out_p, 2, 8, 1e-5), c_rms(ptr, ptr, out_p, 0, 8, 1e-5)) == (-1, -2)
    assert (c_gemm(null, ptr, out_p, 2, 2, 8), c_gemm(ptr, ptr, out_p, 0, 2, 8)) == (-1, -2)
    assert (c_smax(null, out_p, 2, 8), c_smax(ptr, out_p, 0, 8)) == (-1, -2)
    exp = 0 if supported else -5
    assert (c_rms(ptr, ptr, out_p, 2, 8, 1e-5), c_gemm(ptr, ptr, out_p, 2, 2, 2)) == (exp, exp)
    assert c_smax(ptr, out_p, 2, 8) == exp
    if not supported:
        with pytest.raises(RuntimeError, match="not supported"):
            gemm_avx2(buf, buf.T)
        with pytest.raises(RuntimeError, match="not supported"):
            softmax_avx2(buf)


def test_avx2_rmsnorm_compiles_without_mavx2(tmp_path: Path) -> None:
    csrc = Path(avx2_rmsnorm.__file__).resolve().parents[1] / "csrc" / "avx2_rmsnorm.c"
    cc = shutil.which("clang") or shutil.which("cc")
    assert cc is not None
    target = ["--target=x86_64-apple-macos"] if sys.platform == "darwin" else []
    res = subprocess.run([cc, *target, "-O3", "-fPIC", "-c", str(csrc), "-o", str(tmp_path / "r.o")], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
