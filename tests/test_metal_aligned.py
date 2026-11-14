import numpy as np
from parallax.metal.aligned import allocate_page_aligned, free_page_aligned

# Define the page size in bytes (e.g., 4KB)

def test_allocate_page_aligned():
    # Allocate a buffer of size 1024 bytes
    buffer = allocate_page_aligned(1024)
    
    # Check if the buffer is page-aligned
    assert buffer.ctypes.data % PAGE_SIZE == 0, "Buffer is not page-aligned"

    # Free the buffer
    free_page_aligned(buffer)

def test_free_page_aligned():
    # Allocate a buffer of size 1024 bytes
    buffer = allocate_page_aligned(1024)
    
    # Check if the buffer is still accessible after freeing it
    assert buffer.ctypes.data % PAGE_SIZE == 0, "Buffer is not page-aligned"

def test_free_page_aligned_with_large_buffer():
    # Allocate a buffer of size 10MB
