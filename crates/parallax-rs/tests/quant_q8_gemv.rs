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
