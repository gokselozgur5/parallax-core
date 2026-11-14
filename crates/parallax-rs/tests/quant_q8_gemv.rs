use parallax_rs::{gemv_q8_0, vec_dot_q8_0_f32, BlockQ8_0};

#[test]
fn test_block_q8_0_quantize_dequantize_roundtrip() {
    let mut input = [0.0f32; 32];
    for i in 0..32 {
        input[i] = (i as f32) - 16.0;
    }
    let block = BlockQ8_0::quantize(&input);
    let mut out = [0.0f32; 32];
    block.dequantize(&mut out);
    for i in 0..32 {
        assert!((input[i] - out[i]).abs() <= 0.2);
    }
}

#[test]
fn test_gemv_q8_0_identity_mapping() {
    let rows = 2;
    let cols = 32;
    let mut a_vals = [0.0f32; 32];
    a_vals[0] = 1.0;
    let block0 = BlockQ8_0::quantize(&a_vals);
    let mut b_vals = [0.0f32; 32];
    b_vals[1] = 1.0;
    let block1 = BlockQ8_0::quantize(&b_vals);

    let blocks = [block0, block1];
    let mut x = [0.0f32; 32];
    x[0] = 3.0;
    x[1] = 5.0;

    let mut y = [0.0f32; 2];
    gemv_q8_0(rows, cols, &blocks, &x, &mut y);
    assert!((y[0] - 3.0).abs() < 1e-3);
    assert!((y[1] - 5.0).abs() < 1e-3);
}
