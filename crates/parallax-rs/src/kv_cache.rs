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
