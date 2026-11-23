"""Tests for Q8_0 GEMV kernels and C-ABI layout compatibility."""

import ctypes
import numpy as np


class CtypesBlockQ8_0(ctypes.Structure):
    _fields_ = [
        ("scale", ctypes.c_float),
        ("qs", ctypes.c_int8 * 32),
    ]


def test_block_q8_0_ctypes_layout():
    assert ctypes.sizeof(CtypesBlockQ8_0) == 36
    assert CtypesBlockQ8_0.scale.offset == 0
    assert CtypesBlockQ8_0.qs.offset == 4


def test_block_q8_0_ctypes_structure_packing():
    block = CtypesBlockQ8_0()
    block.scale = 2.5
    for i in range(32):
        block.qs[i] = i - 16

    raw = bytes(block)
    assert len(raw) == 36
    scale_from_raw = np.frombuffer(raw[:4], dtype=np.float32)[0]
    qs_from_raw = np.frombuffer(raw[4:], dtype=np.int8)
    assert abs(scale_from_raw - 2.5) < 1e-6
    np.testing.assert_array_equal(qs_from_raw, np.arange(-16, 16, dtype=np.int8))
