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
    if slice.is_empty() {
        return None;
    }
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
pub fn check_target_len(
    logits: &[f32],
    gamma: usize,
    vocab_size: usize,
) -> Result<(), SpeculativeError> {
    let exp = (gamma + 1)
        .checked_mul(vocab_size)
        .ok_or(SpeculativeError::DimensionMismatch)?;
    if logits.len() != exp {
        Err(SpeculativeError::DimensionMismatch)
    } else {
        Ok(())
    }
}

/// Greedy speculative verification matching draft tokens against target logits.
pub fn verify_draft_greedy(
    target_logits: &[f32],
    draft_tokens: &[u32],
    gamma: usize,
    vocab_size: usize,
) -> Result<VerificationResult, SpeculativeError> {
    if draft_tokens.len() != gamma {
        return Err(SpeculativeError::LengthMismatch);
    }
    check_target_len(target_logits, gamma, vocab_size)?;
    let mut acc = Vec::with_capacity(gamma);
    let mut all_acc = true;
    for (i, &cand) in draft_tokens.iter().enumerate() {
        let row = &target_logits[i * vocab_size..(i + 1) * vocab_size];
        let best = argmax_f32(row).ok_or(SpeculativeError::EmptyVocab)? as u32;
        if cand == best {
            acc.push(cand);
        } else {
            acc.push(best);
            all_acc = false;
            break;
        }
    }
    let bonus = if all_acc {
        let b_row = &target_logits[gamma * vocab_size..(gamma + 1) * vocab_size];
        Some(argmax_f32(b_row).ok_or(SpeculativeError::EmptyVocab)? as u32)
    } else {
        None
    };
    let num_acc = if all_acc { acc.len() } else { acc.len() - 1 };
    Ok(VerificationResult {
        accepted_tokens: acc,
        bonus_token: bonus,
        num_accepted: num_acc,
        all_accepted: all_acc,
    })
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
    if ratio.is_nan() {
        return false;
    }
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
    if probs.is_empty() {
        return Err(SpeculativeError::EmptyVocab);
    }
    let mut acc = 0.0f32;
    for (idx, &p) in probs.iter().enumerate() {
        acc += p;
        if roll < acc {
            return Ok(idx as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

/// Calculates acceptance statistics (rate, accepted tokens, total proposed).
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct AcceptanceStats {
    pub total_proposed: usize,
    pub total_accepted: usize,
    pub acceptance_rate: f32,
}

impl AcceptanceStats {
    pub fn compute(proposed: usize, accepted: usize) -> Self {
        let rate = if proposed > 0 { accepted as f32 / proposed as f32 } else { 0.0 };
        Self { total_proposed: proposed, total_accepted: accepted, acceptance_rate: rate }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_f32_basic() {
        let logits = [-1.0f32, 2.5, 0.3, -4.0];
        assert_eq!(argmax_f32(&logits), Some(1));
        assert_eq!(argmax_f32(&[]), None);
    }

    #[test]
    fn test_verify_draft_greedy_all_accepted() {
        let target_logits = [0.1f32, 0.9, 0.2, 0.8, 0.1, 0.3, 0.2, 0.3, 0.7];
        let draft_tokens = [1u32, 0];
        let res = verify_draft_greedy(&target_logits, &draft_tokens, 2, 3).unwrap();
        assert_eq!(res.num_accepted, 2);
        assert!(res.all_accepted);
        assert_eq!(res.accepted_tokens, vec![1, 0]);
        assert_eq!(res.bonus_token, Some(2));
    }

    #[test]
    fn test_verify_draft_greedy_partial_rejection() {
        let target_logits = [0.1f32, 0.9, 0.2, 0.1, 0.8, 0.1, 0.0, 0.0, 1.0];
        let draft_tokens = [1u32, 0];
        let res = verify_draft_greedy(&target_logits, &draft_tokens, 2, 3).unwrap();
        assert_eq!(res.num_accepted, 1);
        assert!(!res.all_accepted);
        assert_eq!(res.accepted_tokens, vec![1, 1]);
        assert_eq!(res.bonus_token, None);
    }

    #[test]
    fn test_verify_candidate_stochastic_and_residual() {
        let p_target = [0.2f32, 0.5, 0.3];
        let p_draft = [0.1f32, 0.6, 0.3];
        assert!(verify_candidate_stochastic(&p_target, 0.6, 1, 0.5));
        assert!(!verify_candidate_stochastic(&p_target, 0.6, 1, 0.9));
        let mut residual = [0.0f32; 3];
        compute_residual_distribution(&p_target, &p_draft, &mut residual).unwrap();
        assert!((residual[0] - 1.0).abs() < 1e-6);
        assert_eq!(residual[1], 0.0);
    }

    #[test]
    fn test_sample_token_and_stats() {
        let probs = [0.1f32, 0.4, 0.5];
        assert_eq!(sample_token_from_probs(&probs, 0.05).unwrap(), 0);
        assert_eq!(sample_token_from_probs(&probs, 0.3).unwrap(), 1);
        assert_eq!(sample_token_from_probs(&probs, 0.8).unwrap(), 2);
        let stats = AcceptanceStats::compute(4, 3);
        assert!((stats.acceptance_rate - 0.75).abs() < 1e-6);
    }
}
