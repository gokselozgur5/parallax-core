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
