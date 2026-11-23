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

    #[inline]
    pub fn to_f32(self) -> f32 {
        let sign = ((self.0 & 0x8000) as u32) << 16;
        let exp = ((self.0 >> 10) & 0x1F) as u32;
        let mant = (self.0 & 0x03FF) as u32;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                let (mut m, mut e) = (mant, 113u32);
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x03FF) << 13)
            }
        } else if exp == 31 {
            sign | 0x7F80_0000 | (mant << 13)
        } else {
            sign | ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
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
fn dot_product_scaled_scalar(q: &[f32], k: &[f32], q_scale: f32, k_scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for (&qi, &ki) in q.iter().zip(k.iter()) {
        sum += (qi * q_scale) * (ki * k_scale);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_scaled_avx512(q: &[f32], k: &[f32], q_scale: f32, k_scale: f32) -> f32 {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_reduce_add_ps, _mm512_set1_ps,
        _mm512_setzero_ps,
    };
    let n = q.len().min(k.len());
    let (vqs, vks) = (_mm512_set1_ps(q_scale), _mm512_set1_ps(k_scale));
    let mut acc = _mm512_setzero_ps();
    let ((qc, qr), (kc, kr)) = (q[..n].as_chunks::<16>(), k[..n].as_chunks::<16>());
    for (qb, kb) in qc.iter().zip(kc.iter()) {
        let vq = _mm512_mul_ps(_mm512_loadu_ps(qb.as_ptr()), vqs);
        let vk = _mm512_mul_ps(_mm512_loadu_ps(kb.as_ptr()), vks);
        acc = _mm512_fmadd_ps(vq, vk, acc);
    }
    _mm512_reduce_add_ps(acc) + dot_product_scaled_scalar(qr, kr, q_scale, k_scale)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_scaled_neon(q: &[f32], k: &[f32], q_scale: f32, k_scale: f32) -> f32 {
    use std::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32};
    let n = q.len().min(k.len());
    let (vqs, vks) = (vdupq_n_f32(q_scale), vdupq_n_f32(k_scale));
    let mut acc = vdupq_n_f32(0.0);
    let ((qc, qr), (kc, kr)) = (q[..n].as_chunks::<4>(), k[..n].as_chunks::<4>());
    for (qb, kb) in qc.iter().zip(kc.iter()) {
        let vq = vmulq_f32(vld1q_f32(qb.as_ptr()), vqs);
        let vk = vmulq_f32(vld1q_f32(kb.as_ptr()), vks);
        acc = vfmaq_f32(acc, vq, vk);
    }
    vaddvq_f32(acc) + dot_product_scaled_scalar(qr, kr, q_scale, k_scale)
}

#[inline]
pub fn dot_product_scaled(q: &[f32], k: &[f32], q_scale: f32, k_scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        return unsafe { dot_product_scaled_avx512(q, k, q_scale, k_scale) };
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { dot_product_scaled_neon(q, k, q_scale, k_scale) };
    }
    dot_product_scaled_scalar(q, k, q_scale, k_scale)
}

#[inline]
fn vec_scale_add_scalar(out: &mut [f32], alpha: f32, v: &[f32], weight: f32) {
    for (o_val, &v_val) in out.iter_mut().zip(v.iter()) {
        *o_val = *o_val * alpha + weight * v_val;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vec_scale_add_avx512(out: &mut [f32], alpha: f32, v: &[f32], weight: f32) {
    use std::arch::x86_64::{
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
    };
    let n = out.len().min(v.len());
    let (va, vw) = (_mm512_set1_ps(alpha), _mm512_set1_ps(weight));
    let ((oc, or), (vc, vr)) = (out[..n].as_chunks_mut::<16>(), v[..n].as_chunks::<16>());
    for (ob, vb) in oc.iter_mut().zip(vc.iter()) {
        let vo = _mm512_mul_ps(_mm512_loadu_ps(ob.as_ptr()), va);
        _mm512_storeu_ps(ob.as_mut_ptr(), _mm512_fmadd_ps(vw, _mm512_loadu_ps(vb.as_ptr()), vo));
    }
    vec_scale_add_scalar(or, alpha, vr, weight);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_scale_add_neon(out: &mut [f32], alpha: f32, v: &[f32], weight: f32) {
    use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    let n = out.len().min(v.len());
    let (va, vw) = (vdupq_n_f32(alpha), vdupq_n_f32(weight));
    let ((oc, or), (vc, vr)) = (out[..n].as_chunks_mut::<4>(), v[..n].as_chunks::<4>());
    for (ob, vb) in oc.iter_mut().zip(vc.iter()) {
        let vo = vmulq_f32(vld1q_f32(ob.as_ptr()), va);
        vst1q_f32(ob.as_mut_ptr(), vfmaq_f32(vo, vw, vld1q_f32(vb.as_ptr())));
    }
    vec_scale_add_scalar(or, alpha, vr, weight);
}

#[inline]
fn vec_scale_add(out: &mut [f32], alpha: f32, v: &[f32], weight: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        unsafe { vec_scale_add_avx512(out, alpha, v, weight) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { vec_scale_add_neon(out, alpha, v, weight) };
        return;
    }
    vec_scale_add_scalar(out, alpha, v, weight);
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
        vec_scale_add(out, alpha, v, 1.0);
    } else {
        let p = (score - *m_i).exp();
        *l_i += p;
        vec_scale_add(out, 1.0, v, p);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vec_scale_avx512(out: &mut [f32], scale: f32) {
    use std::arch::x86_64::{_mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps};
    let vs = _mm512_set1_ps(scale);
    let (oc, or) = out.as_chunks_mut::<16>();
    for ob in oc.iter_mut() {
        _mm512_storeu_ps(ob.as_mut_ptr(), _mm512_mul_ps(_mm512_loadu_ps(ob.as_ptr()), vs));
    }
    for val in or.iter_mut() {
        *val *= scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_scale_neon(out: &mut [f32], scale: f32) {
    use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32};
    let vs = vdupq_n_f32(scale);
    let (oc, or) = out.as_chunks_mut::<4>();
    for ob in oc.iter_mut() {
        vst1q_f32(ob.as_mut_ptr(), vmulq_f32(vld1q_f32(ob.as_ptr()), vs));
    }
    for val in or.iter_mut() {
        *val *= scale;
    }
}

#[inline]
pub fn normalize_output_row(out: &mut [f32], l_i: f32) {
    if l_i > 0.0 {
        let inv_l = 1.0 / l_i;
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            unsafe { vec_scale_avx512(out, inv_l) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { vec_scale_neon(out, inv_l) };
            return;
        }
        for val in out.iter_mut() {
            *val *= inv_l;
        }
    }
}

pub fn validate_attention_dims<T>(
    seq_len: usize,
    head_dim: usize,
    q: &[T],
    k: &[T],
    v: &[T],
    out: &[T],
) -> Result<(), FlashAttnError> {
    let req_len = seq_len
        .checked_mul(head_dim)
        .ok_or(FlashAttnError::DimensionOverflow)?;
    if q.len() != req_len || k.len() != req_len || v.len() != req_len || out.len() != req_len {
        return Err(FlashAttnError::DimensionMismatch);
    }
    Ok(())
}

#[inline]
pub fn dot_product_f16_prescaled(q: &[f32], k: &[f32], q_scale: f32, k_scale: f32) -> f32 {
    let n = q.len().min(k.len());
    let (mut sum, mut offset) = (0.0f32, 0usize);
    let (mut q_tile, mut k_tile) = ([0.0f32; 16], [0.0f32; 16]);
    while offset < n {
        let len = (n - offset).min(16);
        for ((qt, &qi), (kt, &ki)) in q_tile[..len]
            .iter_mut()
            .zip(&q[offset..offset + len])
            .zip(k_tile[..len].iter_mut().zip(&k[offset..offset + len]))
        {
            *qt = F16::from_f32_saturating(qi * q_scale).to_f32();
            *kt = F16::from_f32_saturating(ki * k_scale).to_f32();
        }
        sum += dot_product_scaled(&q_tile[..len], &k_tile[..len], 1.0, 1.0);
        offset += len;
    }
    sum
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
            let s = if config.f16_scale_correction {
                dot_product_f16_prescaled(qi, kj, q_scale, k_scale)
            } else {
                dot_product_scaled(qi, kj, q_scale, k_scale)
            };
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

#[inline]
pub fn dot_product_f16_scaled(q: &[F16], k: &[F16], scale: f32, f16_correction: bool) -> f32 {
    if f16_correction {
        let (qs, ks) = compute_scale_factor(scale, true);
        let (n, mut sum, mut off) = (q.len().min(k.len()), 0.0f32, 0usize);
        let (mut qt, mut kt) = ([0.0f32; 16], [0.0f32; 16]);
        while off < n {
            let len = (n - off).min(16);
            for ((qd, &qi), (kd, &ki)) in qt[..len].iter_mut().zip(&q[off..off + len]).zip(kt[..len].iter_mut().zip(&k[off..off + len])) {
                *qd = F16::from_f32_saturating(qi.to_f32() * qs).to_f32();
                *kd = F16::from_f32_saturating(ki.to_f32() * ks).to_f32();
            }
            sum += dot_product_scaled(&qt[..len], &kt[..len], 1.0, 1.0);
            off += len;
        }
        sum
    } else {
        q.iter().zip(k.iter()).map(|(&qi, &ki)| F16::from_f32(qi.to_f32() * ki.to_f32()).to_f32()).sum::<f32>() * scale
    }
}

pub fn flash_attn_f16(
    q: &[F16], k: &[F16], v: &[F16], out: &mut [F16],
    seq_len: usize, head_dim: usize, config: FlashAttnConfig,
) -> Result<(), FlashAttnError> {
    if config.scale < 0.0 || config.scale.is_nan() { return Err(FlashAttnError::InvalidScale); }
    validate_attention_dims(seq_len, head_dim, q, k, v, out)?;
    let (mut oi, mut vj) = (vec![0.0f32; head_dim], vec![0.0f32; head_dim]);
    for i in 0..seq_len {
        let qi = &q[i * head_dim..(i + 1) * head_dim];
        oi.fill(0.0);
        let (mut m_i, mut l_i) = (f32::NEG_INFINITY, 0.0f32);
        let max_j = if config.is_causal { i + 1 } else { seq_len };
        for j in 0..max_j {
            let kj = &k[j * head_dim..(j + 1) * head_dim];
            for (vd, &vs) in vj.iter_mut().zip(&v[j * head_dim..(j + 1) * head_dim]) { *vd = vs.to_f32(); }
            update_kv_step(&mut oi, &vj, dot_product_f16_scaled(qi, kj, config.scale, config.f16_scale_correction), &mut m_i, &mut l_i);
        }
        normalize_output_row(&mut oi, l_i);
        for (od, &os) in out[i * head_dim..(i + 1) * head_dim].iter_mut().zip(&oi) { *od = F16::from_f32_saturating(os); }
    }
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

    #[test]
    fn test_f16_roundtrip_and_saturation() {
        assert_eq!(F16::ZERO.to_f32(), 0.0);
        assert_eq!(F16::from_f32(1.0).to_f32(), 1.0);
        assert_eq!(F16::from_f32(-2.5).to_f32(), -2.5);
        assert!((F16::from_f32(5.960_464_5e-8).to_f32() - 5.960_464_5e-8).abs() < 1e-12);
        assert_eq!(F16::from_f32(F16_MAX).to_f32(), F16_MAX);
        assert!(F16::from_f32(70_000.0).to_f32().is_infinite());
        assert_eq!(F16::from_f32_saturating(70_000.0).to_f32(), F16_MAX);
        assert_eq!(F16::from_f32_saturating(-70_000.0).to_f32(), -F16_MAX);
        assert!(F16::from_f32_saturating(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn test_flash_attn_f16_overflow_vs_scale_correction() {
        let q = [F16::from_f32(300.0); 4];
        let k = [F16::from_f32(300.0), F16::from_f32(300.0), F16::from_f32(250.0), F16::from_f32(250.0)];
        let v = [F16::from_f32(2.0), F16::from_f32(4.0), F16::from_f32(6.0), F16::from_f32(8.0)];
        assert!(dot_product_f16_scaled(&q[..2], &k[..2], 0.0625, false).is_infinite());
        assert!((dot_product_f16_scaled(&q[..2], &k[..2], 0.0625, true) - 11_250.0).abs() < 1.0);
        let mut out = [F16::ZERO; 4];
        let cfg = FlashAttnConfig { scale: 0.0625, is_causal: false, f16_scale_correction: true };
        assert_eq!(flash_attn_f16(&q, &k, &v, &mut out, 2, 2, cfg), Ok(()));
        assert!((out[0].to_f32() - 2.0).abs() < 1e-3 && (out[1].to_f32() - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_flash_attn_simd_multilane_with_tail() {
        let (seq_len, d) = (2, 35);
        let q: Vec<f32> = (0..seq_len * d).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
        let k: Vec<f32> = (0..seq_len * d).map(|i| ((i % 5) as f32 - 2.0) * 0.3).collect();
        let v: Vec<f32> = (0..seq_len * d).map(|i| (i as f32) * 0.1).collect();
        let (mut out_std, mut out_corr) = (vec![0.0f32; seq_len * d], vec![0.0f32; seq_len * d]);
        assert_eq!(flash_attn(&q, &k, &v, &mut out_std, seq_len, d, FlashAttnConfig::new(d, false, false)), Ok(()));
        assert_eq!(flash_attn(&q, &k, &v, &mut out_corr, seq_len, d, FlashAttnConfig::new(d, false, true)), Ok(()));
        for (&a, &b) in out_std.iter().zip(out_corr.iter()) {
            assert!((a - b).abs() < 2e-3);
        }
    }
}
