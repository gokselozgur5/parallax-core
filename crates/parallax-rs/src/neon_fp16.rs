//! Scalar FP16 dot-product kernel alignment verification and execution.

/// Byte alignment for 128-bit vector memory operations.
pub const FP16_VECTOR_ALIGNMENT_BYTES: usize = 16;
pub const NEON_SIMD_ALIGNMENT_BYTES: usize = FP16_VECTOR_ALIGNMENT_BYTES;

/// Minimum alignment requirement for individual half-precision float primitives.
pub const FP16_ELEMENT_ALIGNMENT_BYTES: usize = 2;

/// Errors that can occur during scalar FP16 dot-product operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeonFp16Error {
    NullPointer,
    UnalignedPointer,
    DimensionMismatch,
    PointerAliasing,
}

impl std::fmt::Display for NeonFp16Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullPointer => write!(f, "null pointer provided"),
            Self::UnalignedPointer => write!(f, "pointer is not aligned to required boundary"),
            Self::DimensionMismatch => write!(f, "vector length dimension mismatch"),
            Self::PointerAliasing => write!(f, "source and destination pointers alias"),
        }
    }
}

impl std::error::Error for NeonFp16Error {}

/// Checks if a memory address satisfies the required byte alignment.
#[inline(always)]
pub fn is_pointer_aligned<T>(ptr: *const T, align_bytes: usize) -> bool {
    if ptr.is_null() || align_bytes == 0 {
        return false;
    }
    (ptr as usize).is_multiple_of(align_bytes)
}

/// Verifies memory alignment for FP16 vector operands.
pub fn verify_fp16_alignment(
    a_ptr: *const u16,
    b_ptr: *const u16,
    strict_simd: bool,
) -> Result<(), NeonFp16Error> {
    if a_ptr.is_null() || b_ptr.is_null() {
        return Err(NeonFp16Error::NullPointer);
    }
    let required = if strict_simd {
        FP16_VECTOR_ALIGNMENT_BYTES
    } else {
        FP16_ELEMENT_ALIGNMENT_BYTES
    };
    if !is_pointer_aligned(a_ptr, required) || !is_pointer_aligned(b_ptr, required) {
        return Err(NeonFp16Error::UnalignedPointer);
    }
    Ok(())
}

/// Decodes an IEEE 754 binary16 half-precision float to standard IEEE 754 f32.
#[inline(always)]
pub fn fp16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let val = (mant as f32 / 1024.0) * (2.0f32).powi(-14);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let val = (1.0 + (mant as f32 / 1024.0)) * (2.0f32).powi((exp as i32) - 15);
        if sign == 1 {
            -val
        } else {
            val
        }
    }
}

/// Computes scalar fallback dot-product for unaligned or scalar tail segments.
pub fn dot_fp16_scalar(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    if a.len() != b.len() {
        return Err(NeonFp16Error::DimensionMismatch);
    }
    let mut sum: f32 = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += fp16_to_f32(x) * fp16_to_f32(y);
    }
    Ok(sum)
}

/// Computes FP16 dot-product with runtime alignment verification.
pub fn dot_fp16_verified(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    if a.len() != b.len() {
        return Err(NeonFp16Error::DimensionMismatch);
    }
    verify_fp16_alignment(a.as_ptr(), b.as_ptr(), false)?;
    dot_fp16_scalar(a, b)
}

/// Computes 16-byte aligned scalar FP16 dot product with boundary checking.
pub fn dot_fp16_aligned(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    if a.len() != b.len() {
        return Err(NeonFp16Error::DimensionMismatch);
    }
    verify_fp16_alignment(a.as_ptr(), b.as_ptr(), true)?;
    dot_fp16_scalar(a, b)
}

/// Backwards-compatible alias for dot_fp16_aligned.
#[inline]
pub fn dot_fp16_simd_aligned(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    dot_fp16_aligned(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pointer_aligned() {
        let val: [u8; 32] = [0; 32];
        let p = val.as_ptr();
        assert!(is_pointer_aligned(p, 1));
        assert!(!is_pointer_aligned(std::ptr::null::<u8>(), 16));
        assert!(!is_pointer_aligned(p, 0));
    }

    #[test]
    fn test_verify_fp16_alignment() {
        let buf = [0u16; 16];
        assert_eq!(
            verify_fp16_alignment(std::ptr::null(), buf.as_ptr(), false),
            Err(NeonFp16Error::NullPointer)
        );
        assert_eq!(
            verify_fp16_alignment(buf.as_ptr(), std::ptr::null(), false),
            Err(NeonFp16Error::NullPointer)
        );
        assert!(verify_fp16_alignment(buf.as_ptr(), buf.as_ptr(), false).is_ok());
    }

    #[test]
    fn test_fp16_to_f32_known_values() {
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        assert_eq!(fp16_to_f32(0x3C00), 1.0);
        assert_eq!(fp16_to_f32(0xC000), -2.0);
        assert!(fp16_to_f32(0x7C00).is_infinite());
        assert!(fp16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn test_dot_fp16_scalar_and_dimension_mismatch() {
        let a = [0x3C00, 0x4000]; // 1.0, 2.0
        let b = [0x4200, 0x4400]; // 3.0, 4.0
        let res = dot_fp16_scalar(&a, &b).unwrap();
        assert!((res - 11.0).abs() < 1e-4);
        assert_eq!(dot_fp16_scalar(&a[..1], &b), Err(NeonFp16Error::DimensionMismatch));
    }

    #[test]
    fn test_dot_fp16_verified_parity() {
        let a = [0x3C00; 8];
        let b = [0x3C00; 8];
        let res = dot_fp16_verified(&a, &b).unwrap();
        assert!((res - 8.0).abs() < 1e-4);
    }
}
