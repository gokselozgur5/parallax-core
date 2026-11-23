import ctypes
import threading
from pathlib import Path
import numpy as np
import pytest
import parallax.rust_engine as re
from parallax.rust_engine import rust_gemm


def test_rust_gemm_does_not_round_int64_product_to_float32() -> None:
    a = np.array([[2**24 + 1]], dtype=np.int64)
    b = np.array([[1]], dtype=np.int64)
    try:
        out = rust_gemm(a, b)
    except TypeError:
        return
    assert int(out[0, 0]) == 16777217


def test_rust_gemm_complex_and_non_float32_buffer_rejected() -> None:
    with pytest.raises(TypeError):
        rust_gemm(np.array([[1 + 2j]]), np.array([[1 + 2j]]))
    with pytest.raises(TypeError):
        rust_gemm(np.zeros((2, 2)), np.zeros((2, 2)), c=np.zeros((2, 2), dtype=np.int64))


def test_require_rust_engine_raises_when_library_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PARALLAX_RUST_LIB", str(tmp_path / "missing.dylib"))
    monkeypatch.setenv("PARALLAX_REQUIRE_RUST_ENGINE", "1")
    re.reset_rust_lib_cache()
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError):
                re.is_rust_engine_available()
        with pytest.raises(RuntimeError):
            re.rust_gemm(np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32))
    finally:
        re.reset_rust_lib_cache()


def test_get_rust_lib_sets_attempted_only_after_handle_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (fake_path := tmp_path / "libparallax_rs.dylib").write_text("stub")
    entered, release = threading.Event(), threading.Event()
    fn = {"__getattr__": lambda s, k: (lambda *_: 3 if k == "parallax_version" else 0)}

    def slow_cdll(_path: str) -> object:
        entered.set()
        release.wait(timeout=5.0)
        return type("FakeCDLL", (), fn)()

    re.reset_rust_lib_cache()
    monkeypatch.setenv("PARALLAX_RUST_LIB", str(fake_path))
    monkeypatch.setattr(ctypes, "CDLL", slow_cdll)
    out: list[object] = []
    ts = [threading.Thread(target=lambda: out.append(re.get_rust_lib())) for _ in range(2)]
    try:
        ts[0].start()
        assert entered.wait(timeout=5.0) and re._LIB_LOAD_ATTEMPTED is False
        ts[1].start()
        release.set()
        ts[0].join(5.0)
        ts[1].join(5.0)
        assert len(out) == 2 and out[0] is not None and out[1] is out[0] and re._LIB_LOAD_ATTEMPTED
    finally:
        release.set()
        ts[0].join(5.0)
        re.reset_rust_lib_cache()
