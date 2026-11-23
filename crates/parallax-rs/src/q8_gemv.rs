//! Quantized Q8_0 dot product and GEMV kernels.

/// A block of 32 quantized 8-bit integers with a fp16/f32 scale.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    pub const BLOCK_SIZE: usize = 32;

    pub fn new(scale: f32, qs: [i8; 32]) -> Self {
        Self { scale, qs }
    }
}

impl BlockQ8_0 {
    pub fn quantize(vals: &[f32; 32]) -> Self {
        let max_abs = vals.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let scale = max_abs / 127.0;
        let mut qs = [0i8; 32];
        if scale > 0.0 {
            for (i, &v) in vals.iter().enumerate() {
                let q = (v / scale).round().clamp(-128.0, 127.0);
                qs[i] = q as i8;
            }
        }
        Self { scale, qs }
    }
}

impl BlockQ8_0 {
    pub fn dequantize(&self, out: &mut [f32; 32]) {
        for (i, &q) in self.qs.iter().enumerate() {
            out[i] = (q as f32) * self.scale;
        }
    }
}

/// Computes dot product between a Q8_0 block and an unquantized f32 slice.
pub fn vec_dot_q8_0_f32_block(block: &BlockQ8_0, x: &[f32; 32]) -> f32 {
    let mut sum_i = 0.0f32;
    for (i, &q) in block.qs.iter().enumerate() {
        sum_i += (q as f32) * x[i];
    }
    sum_i * block.scale
}

/// Computes dot product between a row of Q8_0 blocks and f32 vector x.
pub fn vec_dot_q8_0_f32(blocks: &[BlockQ8_0], x: &[f32]) -> f32 {
    assert_eq!(blocks.len() * 32, x.len());
    let mut sum = 0.0f32;
    for (b_idx, block) in blocks.iter().enumerate() {
        let chunk: &[f32; 32] = x[b_idx * 32..(b_idx + 1) * 32].try_into().unwrap();
        sum += vec_dot_q8_0_f32_block(block, chunk);
    }
    sum
}

/// Matrix-vector product: y = A * x where A is in row-major Q8_0 format.
pub fn gemv_q8_0(
    rows: usize,
    cols: usize,
    a_blocks: &[BlockQ8_0],
    x: &[f32],
    y: &mut [f32],
) {
    let blocks_per_row = cols / 32;
    assert_eq!(cols % 32, 0);
    assert_eq!(a_blocks.len(), rows * blocks_per_row);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    for r in 0..rows {
        let row_slice = &a_blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
        y[r] = vec_dot_q8_0_f32(row_slice, x);
    }
}
