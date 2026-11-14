from __future__ import annotations

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
    return (arr.ctypes.data % ps == 0) and (arr.nbytes % ps == 0)


def aligned_empty(
    shape: int | tuple[int, ...],
    dtype: Any = np.float32,
    page_size: Optional[int] = None,
) -> np.ndarray:
    ps = page_size if page_size is not None else get_page_size()
    dt = np.dtype(dtype)
    num_items = shape if isinstance(shape, int) else int(np.prod(shape))
    alloc_bytes = align_size(max(1, num_items * dt.itemsize), ps)
    total_bytes = alloc_bytes + ps
    mm = mmap.mmap(-1, total_bytes)
    buf = (ctypes.c_char * total_bytes).from_buffer(mm)
    offset = (ps - (ctypes.addressof(buf) % ps)) % ps
    arr = np.frombuffer(
        mm, dtype=dt, count=alloc_bytes // dt.itemsize, offset=offset
    )
    if isinstance(shape, tuple) and num_items * dt.itemsize == alloc_bytes:
        return arr.reshape(shape)
    return arr
