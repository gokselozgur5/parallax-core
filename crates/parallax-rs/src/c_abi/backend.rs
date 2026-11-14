//! C-ABI backend enumeration and device types.

use crate::{STATUS_NOT_FOUND, STATUS_NULL_POINTER, STATUS_OK};

/// Device type identifier for CPU backend.
pub const DEVICE_TYPE_CPU: u32 = 0;
/// Device type identifier for CUDA backend.
pub const DEVICE_TYPE_CUDA: u32 = 1;
/// Device type identifier for Metal backend.
pub const DEVICE_TYPE_METAL: u32 = 2;
/// Device type identifier for ARM NEON backend.
pub const DEVICE_TYPE_NEON: u32 = 3;

/// Returns total number of registered compute backends.
#[no_mangle]
pub extern "C" fn parallax_backend_count() -> u32 {
    4
}

/// Returns device type for the given backend index.
///
/// # Safety
/// `out_device_type` must be non-null and point to valid writable u32.
#[no_mangle]
pub unsafe extern "C" fn parallax_get_device_type(
    backend_idx: u32,
    out_device_type: *mut u32,
) -> i32 {
    if out_device_type.is_null() {
        return STATUS_NULL_POINTER;
    }
    match backend_idx {
        0 => { *out_device_type = DEVICE_TYPE_CPU; STATUS_OK }
        1 => { *out_device_type = DEVICE_TYPE_CUDA; STATUS_OK }
        2 => { *out_device_type = DEVICE_TYPE_METAL; STATUS_OK }
        3 => { *out_device_type = DEVICE_TYPE_NEON; STATUS_OK }
        _ => STATUS_NOT_FOUND,
    }
}
