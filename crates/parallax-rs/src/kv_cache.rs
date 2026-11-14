//! Circular buffer Key-Value cache implementation with boundary wrap-around safety.

/// Error conditions encountered during circular buffer KV cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheError {
    /// Provided dimension was zero or overflowed memory calculations.
    InvalidDimension,
    /// Provided buffer slice length did not match expected dimensions.
    BufferLengthMismatch,
    /// Requested sequence position was out of available token bounds.
    IndexOutOfBounds,
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "invalid dimension or arithmetic overflow"),
            Self::BufferLengthMismatch => write!(f, "buffer length mismatch for KV cache shape"),
            Self::IndexOutOfBounds => write!(f, "token index out of bounds in circular cache"),
        }
    }
}

impl std::error::Error for KvCacheError {}

/// In-place circular buffer KV cache manager with boundary wrap-around safety.
#[derive(Debug, Clone)]
pub struct CircularKvBuffer {
    /// Number of attention heads for key/value.
    pub num_heads: usize,
    /// Head dimension size.
    pub head_dim: usize,
    /// Maximum circular token capacity.
    pub capacity: usize,
    /// Total lifetime tokens ingested (monotonic counter).
    pub total_inserted: usize,
}

impl CircularKvBuffer {
    /// Create a new circular KV buffer descriptor.
    pub fn new(num_heads: usize, head_dim: usize, capacity: usize) -> Result<Self, KvCacheError> {
        if num_heads == 0 || head_dim == 0 || capacity == 0 {
            return Err(KvCacheError::InvalidDimension);
        }
        let per_token = num_heads.checked_mul(head_dim).ok_or(KvCacheError::InvalidDimension)?;
        let _ = capacity.checked_mul(per_token).ok_or(KvCacheError::InvalidDimension)?;
        Ok(Self {
            num_heads,
            head_dim,
            capacity,
            total_inserted: 0,
        })
    }

    /// Number of valid active tokens currently stored in the circular cache.
    #[inline]
    pub fn active_tokens(&self) -> usize {
        self.total_inserted.min(self.capacity)
    }

    /// Slot index in the circular buffer [0, capacity) where the next token is written.
    #[inline]
    pub fn write_cursor(&self) -> usize {
        self.total_inserted % self.capacity
    }
}
