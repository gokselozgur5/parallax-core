//! C-ABI backend enumeration and device types.

/// Device type identifier for CPU backend.
pub const DEVICE_TYPE_CPU: u32 = 0;
/// Device type identifier for CUDA backend.
pub const DEVICE_TYPE_CUDA: u32 = 1;
/// Device type identifier for Metal backend.
pub const DEVICE_TYPE_METAL: u32 = 2;
/// Device type identifier for ARM NEON backend.
pub const DEVICE_TYPE_NEON: u32 = 3;
