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

/// Sequence length binning policy for batching requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceBinner {
    buckets: Vec<usize>,
}

impl SequenceBinner {
    /// Create a new binner from a list of bucket boundaries.
    pub fn new(boundaries: &[usize]) -> Result<Self, SchedulerError> {
        if boundaries.is_empty() {
            return Err(SchedulerError::EmptyBuckets);
        }
        let mut buckets = Vec::new();
        for &b in boundaries {
            if b == 0 {
                return Err(SchedulerError::ZeroLengthSequence);
            }
            if !buckets.contains(&b) {
                buckets.push(b);
            }
        }
        buckets.sort_unstable();
        Ok(Self { buckets })
    }
}

impl SequenceBinner {
    /// Return slice of configured bucket boundaries.
    #[inline]
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// Find smallest bucket boundary that fits seq_len.
    pub fn find_bucket(&self, seq_len: usize) -> Result<usize, SchedulerError> {
        if seq_len == 0 {
            return Err(SchedulerError::ZeroLengthSequence);
        }
        for &b in &self.buckets {
            if seq_len <= b {
                return Ok(b);
            }
        }
        Err(SchedulerError::ExceedsCapacity)
    }
}
