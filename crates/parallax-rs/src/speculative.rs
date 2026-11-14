//! Speculative decoding verification engine.

/// Error conditions during draft verification.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum SpeculativeError {
    EmptyDraft,
    EmptyVocab,
    LengthMismatch,
    InvalidProbability,
    DimensionMismatch,
}

/// Verification outcome for speculative candidate tokens.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct VerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub bonus_token: Option<u32>,
    pub num_accepted: usize,
    pub all_accepted: bool,
}

/// Finds argmax index over a float slice.
pub fn argmax_f32(slice: &[f32]) -> Option<usize> {
    if slice.is_empty() { return None; }
    let mut best_idx = 0;
    let mut best_val = slice[0];
    for (i, &val) in slice.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Some(best_idx)
}
