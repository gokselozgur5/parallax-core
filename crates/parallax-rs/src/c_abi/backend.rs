//! C-ABI backend enumeration and device types.

use crate::{
    is_aligned_u32, STATUS_INVALID_DIMENSION, STATUS_NOT_FOUND, STATUS_NULL_POINTER, STATUS_OK,
    STATUS_UNALIGNED_POINTER,
};

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
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_device_type.is_null() {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_u32(out_device_type) {
            return STATUS_UNALIGNED_POINTER;
        }
        match backend_idx {
            0 => {
                *out_device_type = DEVICE_TYPE_CPU;
                STATUS_OK
            }
            1 => {
                *out_device_type = DEVICE_TYPE_CUDA;
                STATUS_OK
            }
            2 => {
                *out_device_type = DEVICE_TYPE_METAL;
                STATUS_OK
            }
            3 => {
                *out_device_type = DEVICE_TYPE_NEON;
                STATUS_OK
            }
            _ => STATUS_NOT_FOUND,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallax_backend_count() {
        let count = parallax_backend_count();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_parallax_get_device_type_success() {
        let expected = [
            DEVICE_TYPE_CPU,
            DEVICE_TYPE_CUDA,
            DEVICE_TYPE_METAL,
            DEVICE_TYPE_NEON,
        ];
        for (idx, &dev) in expected.iter().enumerate() {
            let mut out = 999u32;
            let status = unsafe { parallax_get_device_type(idx as u32, &mut out) };
            assert_eq!(status, STATUS_OK);
            assert_eq!(out, dev);
        }
    }

    #[test]
    fn test_parallax_get_device_type_errors() {
        let mut out = 0u32;
        let status_oob = unsafe { parallax_get_device_type(4, &mut out) };
        assert_eq!(status_oob, STATUS_NOT_FOUND);
        let status_null = unsafe { parallax_get_device_type(0, std::ptr::null_mut()) };
        assert_eq!(status_null, STATUS_NULL_POINTER);
        let raw: [u8; 16] = [0; 16];
        let unaligned = unsafe { raw.as_ptr().add(1) as *mut u32 };
        let status_unaligned = unsafe { parallax_get_device_type(0, unaligned) };
        assert_eq!(status_unaligned, STATUS_UNALIGNED_POINTER);
    }
}
