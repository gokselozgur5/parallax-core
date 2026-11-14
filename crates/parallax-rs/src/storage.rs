//! Fast memory-mapped tensor storage serializer and deserializer.
//!
//! Provides zero-copy parsing of binary tensor storage archives.

pub const STORAGE_MAGIC: [u8; 4] = *b"PXTS";
pub const STORAGE_VERSION: u32 = 1;
pub const STORAGE_ALIGNMENT: usize = 64;

/// Supported tensor element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum StorageDType {
    F32 = 1,
    F16 = 2,
    BF16 = 3,
    I32 = 4,
    I8 = 5,
    U8 = 6,
}
