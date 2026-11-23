//! Fast memory-mapped tensor storage serializer and deserializer.
//!
//! Provides zero-copy parsing of binary tensor storage archives.

pub mod telemetry;
pub use telemetry::StorageTelemetry;

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
        Ok(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, count) })
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
        if data[0..4] != STORAGE_MAGIC {
            return Err(StorageError::InvalidMagic);
        }
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != STORAGE_VERSION {
            return Err(StorageError::UnsupportedVersion);
        }
        let tensor_count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let reader = Self {
            data,
            tensor_count,
            header_offset: 16,
        };
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
        let end = offset
            .checked_add(name_len)
            .ok_or(StorageError::OffsetOutOfBounds)?;
        if end > self.data.len() {
            return Err(StorageError::UnexpectedEof);
        }
        let name_bytes = &self.data[offset..end];
        let name = std::str::from_utf8(name_bytes).map_err(|_| StorageError::InvalidNameUtf8)?;
        Ok((name, end))
    }
}

impl<'a> StorageReader<'a> {
    fn read_dims(
        &self,
        mut offset: usize,
        ndim: usize,
    ) -> Result<(Vec<usize>, usize), StorageError> {
        let bytes_needed = ndim.checked_mul(8).ok_or(StorageError::OffsetOutOfBounds)?;
        let end = offset
            .checked_add(bytes_needed)
            .ok_or(StorageError::OffsetOutOfBounds)?;
        if end > self.data.len() {
            return Err(StorageError::UnexpectedEof);
        }
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let next = offset + 8;
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

impl<'a> StorageReader<'a> {
    fn validate_data_slice(&self, shape: &[usize], dtype: StorageDType, data_offset: usize) -> Result<&'a [u8], StorageError> {
        let mut numel = 1usize;
        for &d in shape {
            numel = numel.checked_mul(d).ok_or(StorageError::DimensionOverflow)?;
        }
        let byte_len = numel.checked_mul(dtype.element_size()).ok_or(StorageError::DimensionOverflow)?;
        if !data_offset.is_multiple_of(STORAGE_ALIGNMENT) {
            return Err(StorageError::UnalignedTensorData);
        }
        let data_end = data_offset.checked_add(byte_len).ok_or(StorageError::OffsetOutOfBounds)?;
        if data_end > self.data.len() {
            return Err(StorageError::OffsetOutOfBounds);
        }
        Ok(&self.data[data_offset..data_end])
    }
}

impl<'a> StorageReader<'a> {
    fn read_tensor_entry(&self, offset: usize) -> Result<(TensorView<'a>, usize), StorageError> {
        let (dtype, ndim, name_len, data_offset, offset) = self.parse_entry_header(offset)?;
        let (name, offset) = self.read_name(offset, name_len)?;
        let (shape, offset) = self.read_dims(offset, ndim)?;
        let data = self.validate_data_slice(&shape, dtype, data_offset)?;
        Ok((TensorView { name, dtype, shape, data }, offset))
    }
}

impl<'a> StorageReader<'a> {
    fn validate_archive(&self) -> Result<(), StorageError> {
        let mut offset = self.header_offset;
        let mut last_data_end = 0usize;
        for _ in 0..self.tensor_count {
            let (view, next_offset) = self.read_tensor_entry(offset)?;
            let t_start = view.data.as_ptr() as usize - self.data.as_ptr() as usize;
            let t_end = t_start + view.data.len();
            if t_start < last_data_end {
                return Err(StorageError::OverlappingTensorData);
            }
            last_data_end = t_end;
            offset = next_offset;
        }
        Ok(())
    }
}

impl<'a> StorageReader<'a> {
    pub fn get(&self, tensor_name: &str) -> Result<Option<TensorView<'a>>, StorageError> {
        let mut offset = self.header_offset;
        for _ in 0..self.tensor_count {
            let (view, next_offset) = self.read_tensor_entry(offset)?;
            if view.name == tensor_name {
                return Ok(Some(view));
            }
            offset = next_offset;
        }
        Ok(None)
    }

    pub fn iter<'s>(&'s self) -> StorageIter<'s, 'a> {
        StorageIter { reader: self, offset: self.header_offset, index: 0 }
    }
}

/// Iterator over tensors in a StorageReader.
pub struct StorageIter<'s, 'a> {
    reader: &'s StorageReader<'a>,
    offset: usize,
    index: usize,
}

impl<'s, 'a> Iterator for StorageIter<'s, 'a> {
    type Item = Result<TensorView<'a>, StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.reader.tensor_count {
            return None;
        }
        match self.reader.read_tensor_entry(self.offset) {
            Ok((view, next_offset)) => {
                self.offset = next_offset;
                self.index += 1;
                Some(Ok(view))
            }
            Err(e) => {
                self.index = self.reader.tensor_count;
                Some(Err(e))
            }
        }
    }
}

/// Binary tensor entry builder for serialized archives.
pub struct TensorEntryBuilder<'a> {
    pub name: &'a str,
    pub dtype: StorageDType,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// Serializes a collection of tensor entries into a memory-mapped compliant binary buffer.
pub fn serialize_storage(entries: &[TensorEntryBuilder]) -> Result<Vec<u8>, StorageError> {
    for entry in entries {
        if entry.name.len() > 256 {
            return Err(StorageError::NameTooLong);
        }
        let mut numel = 1usize;
        for &d in &entry.shape {
            if d == 0 {
                return Err(StorageError::ZeroDimension);
            }
            numel = numel.checked_mul(d).ok_or(StorageError::DimensionOverflow)?;
        }
        let byte_len = numel.checked_mul(entry.dtype.element_size()).ok_or(StorageError::DimensionOverflow)?;
        if entry.data.len() != byte_len {
            return Err(StorageError::OffsetOutOfBounds);
        }
    }
    let mut out = Vec::new();
    serialize_storage_into(entries, &mut out)?;
    Ok(out)
}

fn compute_header_offsets(entries: &[TensorEntryBuilder]) -> Result<(usize, Vec<usize>), StorageError> {
    let mut header_size = 16usize; // Magic, version, count, flags
    for entry in entries {
        header_size = header_size.checked_add(20).ok_or(StorageError::DimensionOverflow)?;
        header_size = header_size.checked_add(entry.name.len()).ok_or(StorageError::DimensionOverflow)?;
        let dims_size = entry.shape.len().checked_mul(8).ok_or(StorageError::DimensionOverflow)?;
        header_size = header_size.checked_add(dims_size).ok_or(StorageError::DimensionOverflow)?;
    }
    let mut data_cursor = (header_size + STORAGE_ALIGNMENT - 1) & !(STORAGE_ALIGNMENT - 1);
    let mut data_offsets = Vec::with_capacity(entries.len());
    for entry in entries {
        data_offsets.push(data_cursor);
        let next = data_cursor.checked_add(entry.data.len()).ok_or(StorageError::DimensionOverflow)?;
        data_cursor = (next + STORAGE_ALIGNMENT - 1) & !(STORAGE_ALIGNMENT - 1);
    }
    Ok((data_cursor, data_offsets))
}

fn write_storage_header(entries: &[TensorEntryBuilder], out: &mut Vec<u8>) {
    out.extend_from_slice(&STORAGE_MAGIC);
    out.extend_from_slice(&STORAGE_VERSION.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
}

fn write_entry_metadata(entry: &TensorEntryBuilder, data_offset: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&(entry.dtype as u32).to_le_bytes());
    out.extend_from_slice(&(entry.shape.len() as u32).to_le_bytes());
    out.extend_from_slice(&(entry.name.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data_offset as u64).to_le_bytes());
    out.extend_from_slice(entry.name.as_bytes());
    for &d in &entry.shape {
        out.extend_from_slice(&(d as u64).to_le_bytes());
    }
}

fn serialize_storage_into(entries: &[TensorEntryBuilder], out: &mut Vec<u8>) -> Result<(), StorageError> {
    let (total_size, data_offsets) = compute_header_offsets(entries)?;
    out.reserve(total_size);
    write_storage_header(entries, out);
    for (entry, &offset) in entries.iter().zip(data_offsets.iter()) {
        write_entry_metadata(entry, offset, out);
    }
    for (entry, &offset) in entries.iter().zip(data_offsets.iter()) {
        while out.len() < offset {
            out.push(0);
        }
        out.extend_from_slice(entry.data);
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_roundtrip() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let entries = [TensorEntryBuilder {
            name: "weight",
            dtype: StorageDType::F32,
            shape: vec![2, 2],
            data: bytes,
        }];
        let buf = serialize_storage(&entries).expect("serialize");
        let reader = StorageReader::parse(&buf).expect("parse");
        assert_eq!(reader.tensor_count(), 1);
        let view = reader.get("weight").expect("get").expect("found");
        assert_eq!(view.name, "weight");
        assert_eq!(view.shape, vec![2, 2]);
        assert_eq!(view.as_f32_slice().unwrap(), &data[..]);
    }
}
