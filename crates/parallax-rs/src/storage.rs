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

/// Error conditions for storage deserialization.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StorageError {
    InvalidMagic,
    UnsupportedVersion,
    UnexpectedEof,
    InvalidDType,
    InvalidNameUtf8,
    OffsetOutOfBounds,
    UnalignedTensorData,
    OverlappingTensorData,
    ZeroDimension,
    NameTooLong,
    DimensionOverflow,
}

/// Deserialized view of a single tensor inside memory-mapped storage.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorView<'a> {
    pub name: &'a str,
    pub dtype: StorageDType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> usize {
        self.element_count() * self.dtype.element_size()
    }

    pub fn as_f32_slice(&self) -> Result<&'a [f32], StorageError> {
        if self.dtype != StorageDType::F32 {
            return Err(StorageError::InvalidDType);
        }
        if !crate::is_aligned_f32(self.data.as_ptr()) {
            return Err(StorageError::UnalignedTensorData);
        }
        let count = self.element_count();
        if self.data.len() != count * 4 {
            return Err(StorageError::OffsetOutOfBounds);
        }
        Ok(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f32, count)
        })
    }
}

/// Memory-mapped tensor storage archive parser.
pub struct StorageReader<'a> {
    data: &'a [u8],
    tensor_count: usize,
    header_offset: usize,
}

impl<'a> StorageReader<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, StorageError> {
        if data.len() < 16 {
            return Err(StorageError::UnexpectedEof);
        }
        if &data[0..4] != &STORAGE_MAGIC {
            return Err(StorageError::InvalidMagic);
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != STORAGE_VERSION {
            return Err(StorageError::UnsupportedVersion);
        }
        let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let reader = Self { data, tensor_count, header_offset: 16 };
        reader.validate_archive()?;
        Ok(reader)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

impl<'a> StorageReader<'a> {
    fn read_name(&self, offset: usize, name_len: usize) -> Result<(&'a str, usize), StorageError> {
        if name_len > 256 {
            return Err(StorageError::NameTooLong);
        }
        let end = offset.checked_add(name_len).ok_or(StorageError::OffsetOutOfBounds)?;
        if end > self.data.len() {
            return Err(StorageError::UnexpectedEof);
        }
        let name_bytes = &self.data[offset..end];
        let name = std::str::from_utf8(name_bytes).map_err(|_| StorageError::InvalidNameUtf8)?;
        Ok((name, end))
    }
}

impl<'a> StorageReader<'a> {
    fn read_dims(&self, mut offset: usize, ndim: usize) -> Result<(Vec<usize>, usize), StorageError> {
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let next = offset.checked_add(8).ok_or(StorageError::OffsetOutOfBounds)?;
            if next > self.data.len() {
                return Err(StorageError::UnexpectedEof);
            }
            let dim = u64::from_le_bytes(self.data[offset..next].try_into().unwrap()) as usize;
            if dim == 0 {
                return Err(StorageError::ZeroDimension);
            }
            shape.push(dim);
            offset = next;
        }
        Ok((shape, offset))
    }
}

impl<'a> StorageReader<'a> {
    fn parse_entry_header(&self, offset: usize) -> Result<(StorageDType, usize, usize, usize, usize), StorageError> {
        let header_end = offset.checked_add(20).ok_or(StorageError::OffsetOutOfBounds)?;
        if header_end > self.data.len() {
            return Err(StorageError::UnexpectedEof);
        }
        let dtype_raw = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
        let dtype = StorageDType::from_u32(dtype_raw).ok_or(StorageError::InvalidDType)?;
        let ndim = u32::from_le_bytes(self.data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let name_len = u32::from_le_bytes(self.data[offset + 8..offset + 12].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(self.data[offset + 12..offset + 20].try_into().unwrap()) as usize;
        Ok((dtype, ndim, name_len, data_offset, header_end))
    }
}
