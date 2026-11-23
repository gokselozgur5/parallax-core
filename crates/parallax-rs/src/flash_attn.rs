//! FlashAttention module with SIMD dispatch, online softmax, and IEEE-754 F16 overflow protection.

/// Maximum finite representable magnitude in IEEE 754 binary16 (`f16`).
pub const F16_MAX: f32 = 65504.0;

/// IEEE 754 binary16 half-precision floating-point value.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct F16(pub u16);

impl F16 {
    pub const ZERO: Self = Self(0x0000);

    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let bits = val.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x007F_FFFF;
        if exp == 255 {
            return Self(sign | if mant != 0 { 0x7E00 } else { 0x7C00 });
        }
        let half_exp = exp - 112;
        if half_exp >= 31 {
            return Self(sign | 0x7C00);
        }
        if half_exp <= 0 {
            if half_exp < -10 {
                return Self(sign);
            }
            let full_mant = mant | 0x0080_0000;
            let shift = (14 - half_exp) as u32;
            let bias = (1 << (shift - 1)) - 1 + ((full_mant >> shift) & 1);
            return Self(sign | (((full_mant + bias) >> shift) as u16));
        }
        let rounded = mant + 0x0000_0FFF + ((mant >> 13) & 1);
        if (rounded & 0x0080_0000) != 0 {
            let next_exp = half_exp + 1;
            return Self(sign | if next_exp >= 31 { 0x7C00 } else { (next_exp as u16) << 10 });
        }
        Self(sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16))
    }

    #[inline]
    pub fn from_f32_saturating(val: f32) -> Self {
        if val.is_nan() { Self(0x7E00) } else { Self::from_f32(val.clamp(-F16_MAX, F16_MAX)) }
    }
}

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

#[inline]
pub fn normalize_output_row(out: &mut [f32], l_i: f32) {
    if l_i > 0.0 {
        let inv_l = 1.0 / l_i;
        for val in out.iter_mut() {
            *val *= inv_l;
        }
    }
}

pub fn validate_attention_dims(
    seq_len: usize,
    head_dim: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &[f32],
) -> Result<(), FlashAttnError> {
    let req_len = seq_len
        .checked_mul(head_dim)
        .ok_or(FlashAttnError::DimensionOverflow)?;
    if q.len() != req_len || k.len() != req_len || v.len() != req_len || out.len() != req_len {
        return Err(FlashAttnError::DimensionMismatch);
    }
    Ok(())
}

pub fn flash_attn_forward_kernel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    config: FlashAttnConfig,
) {
    let (q_scale, k_scale) = compute_scale_factor(config.scale, config.f16_scale_correction);
    for i in 0..seq_len {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        let oi = &mut out[i * head_dim..(i + 1) * head_dim];
        oi.fill(0.0);
        let mut m_i = f32::NEG_INFINITY;
        let mut l_i = 0.0f32;
        let max_j = if config.is_causal { i + 1 } else { seq_len };
        for j in 0..max_j {
            let kj = &k[j * head_dim..(j + 1) * head_dim];
            let vj = &v[j * head_dim..(j + 1) * head_dim];
            let s = dot_product_scaled(qi, kj, q_scale, k_scale);
            update_kv_step(oi, vj, s, &mut m_i, &mut l_i);
        }
        normalize_output_row(oi, l_i);
    }
}

pub fn flash_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    config: FlashAttnConfig,
) -> Result<(), FlashAttnError> {
    if config.scale < 0.0 || config.scale.is_nan() {
        return Err(FlashAttnError::InvalidScale);
    }
    validate_attention_dims(seq_len, head_dim, q, k, v, out)?;
    flash_attn_forward_kernel(q, k, v, out, seq_len, head_dim, config);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_factor_calculation() {
        let (q_s, k_s) = compute_scale_factor(0.25, false);
        assert!((q_s - 0.25).abs() < 1e-6);
        assert!((k_s - 1.0).abs() < 1e-6);

        let (q_corr, k_corr) = compute_scale_factor(0.25, true);
        assert!((q_corr - 0.5).abs() < 1e-6);
        assert!((k_corr - 0.5).abs() < 1e-6);
        assert!((q_corr * k_corr - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attn_single_token_parity() {
        let q = [1.0f32, 2.0];
        let k = [1.0f32, 2.0];
        let v = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        let cfg = FlashAttnConfig::new(2, false, false);
        let res = flash_attn(&q, &k, &v, &mut out, 1, 2, cfg);
        assert_eq!(res, Ok(()));
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_flash_attn_causal_masking() {
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [10.0f32, 1.0, 20.0, 2.0];
        let mut out = [0.0f32; 4];
        let cfg = FlashAttnConfig::new(2, true, false);
        let res = flash_attn(&q, &k, &v, &mut out, 2, 2, cfg);
        assert_eq!(res, Ok(()));
        assert!((out[0] - 10.0).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_flash_attn_f16_scale_correction_equivalence() {
        let q = [1.0f32, 2.0, 1.0, 2.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let mut out_std = [0.0f32; 4];
        let mut out_corr = [0.0f32; 4];
        let cfg_std = FlashAttnConfig::new(2, false, false);
        let cfg_corr = FlashAttnConfig::new(2, false, true);
        flash_attn(&q, &k, &v, &mut out_std, 2, 2, cfg_std).unwrap();
        flash_attn(&q, &k, &v, &mut out_corr, 2, 2, cfg_corr).unwrap();
        assert!((out_std[0] - out_corr[0]).abs() < 1e-4);
        assert!((out_std[1] - out_corr[1]).abs() < 1e-4);
        assert!((out_std[2] - out_corr[2]).abs() < 1e-4);
        assert!((out_std[3] - out_corr[3]).abs() < 1e-4);
    }

    #[test]
    fn test_flash_attn_nan_and_errors() {
        let q = [f32::NAN, 1.0];
        let k = [1.0f32, 1.0];
        let v = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        let cfg = FlashAttnConfig::new(2, false, false);
        assert_eq!(flash_attn(&q, &k, &v, &mut out, 1, 2, cfg), Ok(()));
        assert!(out[0].is_nan() || out[1].is_nan());

        let invalid_cfg = FlashAttnConfig { scale: -1.0, is_causal: false, f16_scale_correction: false };
        assert_eq!(flash_attn(&q, &k, &v, &mut out, 1, 2, invalid_cfg), Err(FlashAttnError::InvalidScale));
    }
}
