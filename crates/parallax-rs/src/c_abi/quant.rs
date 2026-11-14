use crate::q8_gemv::BlockQ8_0;
use crate::{STATUS_NULL_POINTER, STATUS_OK};

/// Quantizes 32 f32 values into a BlockQ8_0 struct.
///
/// # Safety
/// Both `input_f32` and `out_block` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn parallax_quantize_q8_0(
    input_f32: *const f32,
    out_block: *mut BlockQ8_0,
) -> i32 {
    if input_f32.is_null() || out_block.is_null() {
        return STATUS_NULL_POINTER;
    }
    let slice: &[f32; 32] = &*(input_f32 as *const [f32; 32]);
    *out_block = BlockQ8_0::quantize(slice);
    STATUS_OK
}

/// Dequantizes a BlockQ8_0 struct back into 32 f32 values.
///
/// # Safety
/// Both `block` and `out_f32` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn parallax_dequantize_q8_0(
    block: *const BlockQ8_0,
    out_f32: *mut f32,
) -> i32 {
    if block.is_null() || out_f32.is_null() {
        return STATUS_NULL_POINTER;
    }
    let out_slice: &mut [f32; 32] = &mut *(out_f32 as *mut [f32; 32]);
    (*block).dequantize(out_slice);
    STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallax_quantize_q8_0_null_ptrs() {
        let dummy = [1.0f32; 32];
        let mut block = BlockQ8_0::new(0.0, [0i8; 32]);
        let res1 = unsafe { parallax_quantize_q8_0(std::ptr::null(), &mut block) };
        assert_eq!(res1, STATUS_NULL_POINTER);
        let res2 = unsafe { parallax_quantize_q8_0(dummy.as_ptr(), std::ptr::null_mut()) };
        assert_eq!(res2, STATUS_NULL_POINTER);
    }

    #[test]
    fn test_parallax_quantize_q8_0_success() {
        let vals = [2.0f32; 32];
        let mut block = BlockQ8_0::new(0.0, [0i8; 32]);
        let res = unsafe { parallax_quantize_q8_0(vals.as_ptr(), &mut block) };
        assert_eq!(res, STATUS_OK);
        assert!((block.scale - (2.0 / 127.0)).abs() < 1e-6);
        assert_eq!(block.qs[0], 127);
    }

    #[test]
    fn test_parallax_dequantize_q8_0_null_ptrs() {
        let block = BlockQ8_0::new(0.0, [0i8; 32]);
        let mut out = [0.0f32; 32];
        let res1 = unsafe { parallax_dequantize_q8_0(std::ptr::null(), out.as_mut_ptr()) };
        assert_eq!(res1, STATUS_NULL_POINTER);
        let res2 = unsafe { parallax_dequantize_q8_0(&block, std::ptr::null_mut()) };
        assert_eq!(res2, STATUS_NULL_POINTER);
    }

    #[test]
    fn test_parallax_dequantize_q8_0_success() {
        let block = BlockQ8_0::new(0.5, [2i8; 32]);
        let mut out = [0.0f32; 32];
        let res = unsafe { parallax_dequantize_q8_0(&block, out.as_mut_ptr()) };
        assert_eq!(res, STATUS_OK);
        for v in out.iter() {
            assert!((v - 1.0f32).abs() < 1e-6);
        }
    }
}

