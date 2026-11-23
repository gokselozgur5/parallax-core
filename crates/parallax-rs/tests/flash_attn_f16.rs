use parallax_rs::flash_attn::{flash_attn, FlashAttnConfig};

#[test]
fn f16_scale_correction_prevents_intermediate_overflow() {
    let seq_len = 2;
    let head_dim = 20;
    let q = [2.0e20f32; 40];
    let mut k = [2.0e20f32; 40];
    k[20..].fill(1.5e20);
    let mut v = [2.0f32; 40];
    v[20..].fill(4.0);

    let mut out = [0.0f32; 40];
    let cfg = FlashAttnConfig::new(head_dim, false, true);
    assert_eq!(flash_attn(&q, &k, &v, &mut out, seq_len, head_dim, cfg), Ok(()));
    assert!(
        out.iter().all(|val| val.is_finite()),
        "f16_scale_correction overflowed to non-finite output: {:?}",
        &out[..4]
    );
    assert!((out[0] - 3.0).abs() < 1e-5);
}
