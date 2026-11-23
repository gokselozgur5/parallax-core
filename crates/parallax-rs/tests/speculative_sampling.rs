use parallax_rs::speculative::{sample_token_from_probs, verify_candidate_stochastic};

#[test]
fn zero_probability_token_is_never_sampled() {
    let probs = [0.0f32, 1.0];
    assert_eq!(sample_token_from_probs(&probs, 0.0), Ok(1));
}

#[test]
fn stochastic_verification_rejects_nan_ratio() {
    let target = [f32::NAN];
    assert!(!verify_candidate_stochastic(&target, 0.5, 0, 0.5));
}
