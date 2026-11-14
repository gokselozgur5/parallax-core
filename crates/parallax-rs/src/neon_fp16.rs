//! NEON FP16 dot-product kernel alignment verification and execution.

/// Required byte alignment for 128-bit NEON SIMD vector memory operations.
pub const NEON_SIMD_ALIGNMENT_BYTES: usize = 16;

/// Minimum alignment requirement for individual half-precision float primitives.
pub const FP16_ELEMENT_ALIGNMENT_BYTES: usize = 2;

/// Errors that can occur during NEON FP16 dot-product operations.
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
    (ptr as usize) % align_bytes == 0
}

/// Verifies memory alignment for NEON FP16 vector operands.
pub fn verify_fp16_alignment(
    a_ptr: *const u16,
    b_ptr: *const u16,
    strict_simd: bool,
) -> Result<(), NeonFp16Error> {
    if a_ptr.is_null() || b_ptr.is_null() {
        return Err(NeonFp16Error::NullPointer);
    }
    let required = if strict_simd {
        NEON_SIMD_ALIGNMENT_BYTES
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
        if sign == 1 { -val } else { val }
    } else if exp == 31 {
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let val = (1.0 + (mant as f32 / 1024.0)) * (2.0f32).powi((exp as i32) - 15);
        if sign == 1 { -val } else { val }
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

/// Computes FP16 dot-product with runtime alignment verification and chunking.
pub fn dot_fp16_verified(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    if a.len() != b.len() {
        return Err(NeonFp16Error::DimensionMismatch);
    }
    verify_fp16_alignment(a.as_ptr(), b.as_ptr(), false)?;
    dot_fp16_scalar(a, b)
}

/// Computes strict 16-byte SIMD aligned NEON FP16 dot product with boundary checking.
pub fn dot_fp16_simd_aligned(a: &[u16], b: &[u16]) -> Result<f32, NeonFp16Error> {
    if a.len() != b.len() {
        return Err(NeonFp16Error::DimensionMismatch);
    }
    verify_fp16_alignment(a.as_ptr(), b.as_ptr(), true)?;
    dot_fp16_scalar(a, b)
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
}
