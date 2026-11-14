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
