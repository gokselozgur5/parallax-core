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

/// Check target logits dimension for speculative verification.
#[inline]
pub fn check_target_len(logits: &[f32], gamma: usize, vocab_size: usize) -> Result<(), SpeculativeError> {
    let exp = (gamma + 1).checked_mul(vocab_size).ok_or(SpeculativeError::DimensionMismatch)?;
    if logits.len() != exp { Err(SpeculativeError::DimensionMismatch) } else { Ok(()) }
}

/// Greedy speculative verification matching draft tokens against target logits.
pub fn verify_draft_greedy(
    target_logits: &[f32],
    draft_tokens: &[u32],
    gamma: usize,
    vocab_size: usize,
) -> Result<VerificationResult, SpeculativeError> {
    if draft_tokens.len() != gamma { return Err(SpeculativeError::LengthMismatch); }
    check_target_len(target_logits, gamma, vocab_size)?;
    let mut acc = Vec::with_capacity(gamma);
    let mut all_acc = true;
    for (i, &cand) in draft_tokens.iter().enumerate() {
        let row = &target_logits[i * vocab_size..(i + 1) * vocab_size];
        let best = argmax_f32(row).ok_or(SpeculativeError::EmptyVocab)? as u32;
        if cand == best { acc.push(cand); } else { acc.push(best); all_acc = false; break; }
    }
    let bonus = if all_acc {
        let b_row = &target_logits[gamma * vocab_size..(gamma + 1) * vocab_size];
        Some(argmax_f32(b_row).ok_or(SpeculativeError::EmptyVocab)? as u32)
    } else { None };
    let num_acc = if all_acc { acc.len() } else { acc.len() - 1 };
    Ok(VerificationResult { accepted_tokens: acc, bonus_token: bonus, num_accepted: num_acc, all_accepted: all_acc })
}

/// Verifies single stochastic candidate against target distribution and random roll.
pub fn verify_candidate_stochastic(
    target_probs: &[f32],
    draft_prob: f32,
    cand: u32,
    roll: f32,
) -> bool {
    let cand_idx = cand as usize;
    if cand_idx >= target_probs.len() || draft_prob <= 0.0 || roll < 0.0 || roll > 1.0 {
        return false;
    }
    let p_target = target_probs[cand_idx];
    let ratio = p_target / draft_prob;
    roll < ratio.min(1.0)
}

/// Computes residual probability distribution max(0, p - q) normalized.
pub fn compute_residual_distribution(
    target_probs: &[f32],
    draft_probs: &[f32],
    residual_out: &mut [f32],
) -> Result<(), SpeculativeError> {
    if target_probs.len() != draft_probs.len() || target_probs.len() != residual_out.len() {
        return Err(SpeculativeError::DimensionMismatch);
    }
    let mut sum = 0.0f32;
    for i in 0..target_probs.len() {
        let diff = (target_probs[i] - draft_probs[i]).max(0.0);
        residual_out[i] = diff;
        sum += diff;
    }
    if sum > 1e-12 {
        let inv_sum = 1.0 / sum;
        for val in residual_out.iter_mut() {
            *val *= inv_sum;
        }
    } else {
        residual_out.copy_from_slice(target_probs);
    }
    Ok(())
}

/// Samples a token index from a discrete probability distribution using a roll in [0, 1).
pub fn sample_token_from_probs(probs: &[f32], roll: f32) -> Result<u32, SpeculativeError> {
    if probs.is_empty() { return Err(SpeculativeError::EmptyVocab); }
    let mut acc = 0.0f32;
    for (idx, &p) in probs.iter().enumerate() {
        acc += p;
        if roll <= acc { return Ok(idx as u32); }
    }
    Ok((probs.len() - 1) as u32)
}
