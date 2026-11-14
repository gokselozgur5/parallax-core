//! Dynamic batch scheduler and sequence length binning.

/// Error variants for batch scheduling and sequence binning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    /// Provided bucket boundaries were empty.
    EmptyBuckets,
    /// Provided sequence length was zero.
    ZeroLengthSequence,
    /// Sequence length exceeded largest available bucket.
    ExceedsCapacity,
    /// Target buffer slice length was insufficient.
    BufferTooSmall,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBuckets => write!(f, "empty bucket boundaries"),
            Self::ZeroLengthSequence => write!(f, "sequence length must be non-zero"),
            Self::ExceedsCapacity => write!(f, "sequence length exceeds largest bucket"),
            Self::BufferTooSmall => write!(f, "target buffer too small"),
        }
    }
}

impl std::error::Error for SchedulerError {}
