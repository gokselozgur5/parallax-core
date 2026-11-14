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

    /// Convert a relative logical token index in [0, active_tokens) to physical slot index.
    pub fn logical_to_physical(&self, logical_idx: usize) -> Result<usize, KvCacheError> {
        let active = self.active_tokens();
        if logical_idx >= active {
            return Err(KvCacheError::IndexOutOfBounds);
        }
        if self.total_inserted < self.capacity {
            Ok(logical_idx)
        } else {
            // Buffer has wrapped: oldest token is at write_cursor()
            let phys = (self.write_cursor() + logical_idx) % self.capacity;
            Ok(phys)
        }
    }

    /// Calculate buffer offset and length for a given (head, physical_slot).
    #[inline]
    pub fn slot_offset(&self, head: usize, slot: usize) -> Result<usize, KvCacheError> {
        if head >= self.num_heads || slot >= self.capacity {
            return Err(KvCacheError::InvalidDimension);
        }
        // Offset: (head * capacity + slot) * head_dim
        let h_offset = head.checked_mul(self.capacity).ok_or(KvCacheError::InvalidDimension)?;
        let slot_offset = h_offset.checked_add(slot).ok_or(KvCacheError::InvalidDimension)?;
        slot_offset.checked_mul(self.head_dim).ok_or(KvCacheError::InvalidDimension)
    }

    /// Total expected buffer capacity in float elements (num_heads * capacity * head_dim).
    #[inline]
    pub fn required_buffer_len(&self) -> Result<usize, KvCacheError> {
        let per_head = self.capacity.checked_mul(self.head_dim).ok_or(KvCacheError::InvalidDimension)?;
        self.num_heads.checked_mul(per_head).ok_or(KvCacheError::InvalidDimension)
    }
}
