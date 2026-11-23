from __future__ import annotations

from collections.abc import Sequence
import ctypes
import mmap
import os
from typing import Any, Optional
import numpy as np


def get_page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return getattr(mmap, "PAGESIZE", 4096)


PAGE_SIZE: int = get_page_size()


def align_size(size: int, page_size: Optional[int] = None) -> int:
    ps = page_size if page_size is not None else get_page_size()
    rem = size % ps
    return size if rem == 0 else size + (ps - rem)


def is_aligned(arr: np.ndarray, page_size: Optional[int] = None) -> bool:
    ps = page_size if page_size is not None else get_page_size()
    if arr.ctypes.data % ps != 0:
        return False
    if arr.nbytes > 0 and arr.nbytes % ps == 0:
        return True
    root = arr
    while isinstance(root.base, np.ndarray):
        root = root.base
    offset = int(arr.ctypes.data) - int(root.ctypes.data)
    return (
        root is not arr
        and root.flags.c_contiguous
        and root.ctypes.data % ps == 0
        and root.nbytes > 0
        and root.nbytes % ps == 0
        and 0 <= offset < root.nbytes
        and root.nbytes - offset >= arr.nbytes
    )


def aligned_empty(
    shape: int | Sequence[int],
    dtype: Any = np.float32,
    page_size: Optional[int] = None,
) -> np.ndarray:
    ps = page_size if page_size is not None else get_page_size()
    dt = np.dtype(dtype)
    target_shape = (
        (int(shape),)
        if isinstance(shape, (int, np.integer))
        else tuple(int(d) for d in shape)
    )
    if any(d < 0 for d in target_shape):
        raise ValueError("negative dimensions are not allowed")
    num_items = int(np.prod(target_shape, dtype=np.intp))
    alloc_bytes = align_size(max(1, num_items * dt.itemsize), ps)
    total_bytes = alloc_bytes + ps
    mm = mmap.mmap(-1, total_bytes)
    buf = (ctypes.c_char * total_bytes).from_buffer(mm)
    offset = (ps - (ctypes.addressof(buf) % ps)) % ps
    backing = np.frombuffer(mm, dtype=np.uint8, count=alloc_bytes, offset=offset)
    return np.ndarray(target_shape, dtype=dt, buffer=backing)


def aligned_zeros(
    shape: int | Sequence[int],
    dtype: Any = np.float32,
    page_size: Optional[int] = None,
) -> np.ndarray:
    arr = aligned_empty(shape, dtype=dtype, page_size=page_size)
    arr.fill(0)
    return arr


empty_aligned = aligned_empty
zeros_aligned = aligned_zeros
