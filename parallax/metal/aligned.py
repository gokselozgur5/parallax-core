import ctypes
import numpy as np

# Define the page size in bytes (e.g., 4KB)

def align_to_page(size):
    """
    Aligns a given size to the nearest page boundary.

    Args:
        size (int): The size to be aligned.

    Returns:
        int: The aligned size.
    """
    return (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

def allocate_page_aligned(size):
    """
    Allocates a page-aligned buffer using NumPy.

    Args:
        size (int): The size of the buffer in bytes.

