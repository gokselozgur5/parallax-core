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

/// Batch descriptor for grouped requests assigned to a common bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledBatch {
    /// Bucket capacity for sequence dimension.
    pub bucket_len: usize,
    /// Indices or IDs of sequences assigned to this batch.
    pub sequence_indices: Vec<usize>,
}

impl SequenceBinner {
    /// Calculate padding waste ratio for sequence lengths in a bucket.
    pub fn padding_waste(&self, lengths: &[usize], bucket: usize) -> f64 {
        if lengths.is_empty() || bucket == 0 {
            return 0.0;
        }
        let total = lengths.len() * bucket;
        let actual: usize = lengths.iter().sum();
        let pad = total.saturating_sub(actual);
        pad as f64 / total as f64
    }
}

impl SequenceBinner {
    /// Schedule sequences into length-binned batches.
    pub fn schedule_batches(
        &self,
        lengths: &[usize],
        max_batch_size: usize,
    ) -> Result<Vec<ScheduledBatch>, SchedulerError> {
        if max_batch_size == 0 {
            return Err(SchedulerError::ZeroLengthSequence);
        }
        let mut batches = Vec::new();
        for &b in &self.buckets {
            let mut curr = Vec::new();
            for (i, &l) in lengths.iter().enumerate() {
                if self.find_bucket(l)? == b {
                    curr.push(i);
                    if curr.len() == max_batch_size {
                        batches.push(ScheduledBatch {
                            bucket_len: b,
                            sequence_indices: std::mem::take(&mut curr),
                        });
                    }
                }
            }
            if !curr.is_empty() {
                batches.push(ScheduledBatch {
                    bucket_len: b,
                    sequence_indices: curr,
                });
            }
        }
        Ok(batches)
    }
}

impl SequenceBinner {
    /// Collate sequence data into target padded buffer with attention mask.
    pub fn collate(
        &self,
        sequences: &[&[u32]],
        pad_token: u32,
        out_tokens: &mut [u32],
        out_mask: &mut [u32],
    ) -> Result<usize, SchedulerError> {
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        let bucket = self.find_bucket(max_len)?;
        let req = sequences
            .len()
            .checked_mul(bucket)
            .ok_or(SchedulerError::BufferTooSmall)?;
        if out_tokens.len() < req || out_mask.len() < req {
            return Err(SchedulerError::BufferTooSmall);
        }
        for (i, seq) in sequences.iter().enumerate() {
            let offset = i * bucket;
            for j in 0..bucket {
                if j < seq.len() {
                    out_tokens[offset + j] = seq[j];
                    out_mask[offset + j] = 1;
                } else {
                    out_tokens[offset + j] = pad_token;
                    out_mask[offset + j] = 0;
                }
            }
        }
        Ok(bucket)
    }
}
