use parallax_rs::storage::StorageReader;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static LARGEST_REQUEST: AtomicUsize = AtomicUsize::new(0);

struct RecordingAllocator;

unsafe impl GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST_REQUEST.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

#[test]
fn parse_does_not_reserve_memory_from_untrusted_ndim() {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"PXTS");
    for word in [1u32, 1, 0, 1, u32::MAX, 0] {
        buf.extend_from_slice(&word.to_le_bytes()); // version, count, flags, dtype, ndim, name_len
    }
    buf.extend_from_slice(&64u64.to_le_bytes()); // data offset
    LARGEST_REQUEST.store(0, Ordering::Relaxed);
    assert!(StorageReader::parse(&buf).is_err());
    let largest = LARGEST_REQUEST.load(Ordering::Relaxed);
    assert!(
        largest < 1 << 20,
        "a 36-byte archive requested {largest} bytes"
    );
}
