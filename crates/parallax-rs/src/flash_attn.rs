//! FlashAttention module with online softmax and F16 scale correction.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlashAttnError {
    DimensionMismatch,
    DimensionOverflow,
    InvalidScale,
}

#[derive(Debug, Clone, Copy)]
pub struct FlashAttnConfig {
    pub scale: f32,
    pub is_causal: bool,
    pub f16_scale_correction: bool,
}

impl FlashAttnConfig {
    pub fn new(head_dim: usize, is_causal: bool, f16_scale_correction: bool) -> Self {
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };
        Self {
            scale,
            is_causal,
            f16_scale_correction,
        }
    }
}
