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

impl StorageDType {
    pub const fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    pub const fn from_u32(val: u32) -> Option<Self> {
        match val {
            1 => Some(Self::F32),
            2 => Some(Self::F16),
            3 => Some(Self::BF16),
            4 => Some(Self::I32),
            5 => Some(Self::I8),
            6 => Some(Self::U8),
            _ => None,
        }
    }
}
