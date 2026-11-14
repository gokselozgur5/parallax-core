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
