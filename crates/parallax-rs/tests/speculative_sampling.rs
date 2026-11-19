use parallax_rs::speculative::sample_token_from_probs;

#[test]
fn zero_probability_token_is_never_sampled() {
    let probs = [0.0f32, 1.0];
    assert_eq!(sample_token_from_probs(&probs, 0.0), Ok(1));
}
