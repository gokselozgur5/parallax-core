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

/// Validates matrix dimensions and returns (gamma, vocab_size).
pub fn validate_draft_matrix(
    logits: &[f32],
    gamma: usize,
    vocab_size: usize,
) -> Result<(), SpeculativeError> {
    if gamma == 0 {
        return Err(SpeculativeError::EmptyDraft);
    }
    if vocab_size == 0 {
        return Err(SpeculativeError::EmptyVocab);
    }
    let expected = gamma
        .checked_mul(vocab_size)
        .ok_or(SpeculativeError::DimensionMismatch)?;
    if logits.len() != expected {
        return Err(SpeculativeError::DimensionMismatch);
    }
    Ok(())
}
