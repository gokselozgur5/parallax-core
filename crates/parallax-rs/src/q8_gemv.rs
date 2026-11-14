//! Quantized Q8_0 dot product and GEMV kernels.

/// A block of 32 quantized 8-bit integers with a fp16/f32 scale.
#[derive(Debug, Clone, Copy, PartialEq)]
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
