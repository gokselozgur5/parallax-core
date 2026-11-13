from __future__ import annotations

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
