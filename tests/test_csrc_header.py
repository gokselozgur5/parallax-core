"""Tests for parallax_simd.h header compilation and status codes."""

import subprocess
import tempfile
from pathlib import Path


def test_parallax_simd_header_compiles_standalone():
    hdr = Path(__file__).parent.parent / "parallax" / "csrc" / "parallax_simd.h"
    assert hdr.is_file()
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w") as f:
        f.write('#include "parallax_simd.h"\nint main(void){return PARALLAX_STATUS_OK;}\n')
        f.flush()
        cmd = ["cc", "-Wall", "-Wextra", "-Werror", f"-I{hdr.parent}", "-o", "/dev/null", f.name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_parallax_simd_header_status_definitions():
    hdr = Path(__file__).parent.parent / "parallax" / "csrc" / "parallax_simd.h"
    content = hdr.read_text(encoding="utf-8")
    for code in ("PARALLAX_STATUS_OK", "PARALLAX_STATUS_NULL_POINTER", "PARALLAX_STATUS_INVALID_DIMENSION"):
        assert code in content
