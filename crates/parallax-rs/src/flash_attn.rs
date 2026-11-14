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

#[inline]
pub fn compute_scale_factor(scale: f32, f16_correction: bool) -> (f32, f32) {
    if f16_correction {
        let q_scale = scale.sqrt();
        let k_scale = scale.sqrt();
        (q_scale, k_scale)
    } else {
        (scale, 1.0)
    }
}

#[inline]
pub fn dot_product_scaled(
    q: &[f32],
    k: &[f32],
    q_scale: f32,
    k_scale: f32,
) -> f32 {
    let mut sum = 0.0f32;
    for (&qi, &ki) in q.iter().zip(k.iter()) {
        sum += (qi * q_scale) * (ki * k_scale);
    }
    sum
}

#[inline]
pub fn update_kv_step(
    out: &mut [f32],
    v: &[f32],
    score: f32,
    m_i: &mut f32,
    l_i: &mut f32,
) {
    if score.is_infinite() && score < 0.0 {
        return;
    }
    if score > *m_i {
        let alpha = (*m_i - score).exp();
        *m_i = score;
        *l_i = *l_i * alpha + 1.0;
        for (o_val, &v_val) in out.iter_mut().zip(v.iter()) {
            *o_val = *o_val * alpha + v_val;
        }
    } else {
        let p = (score - *m_i).exp();
        *l_i += p;
        for (o_val, &v_val) in out.iter_mut().zip(v.iter()) {
            *o_val += p * v_val;
        }
    }
}
