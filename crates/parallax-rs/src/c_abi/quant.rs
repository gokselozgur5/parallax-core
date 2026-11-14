use crate::q8_gemv::BlockQ8_0;
use crate::{STATUS_NULL_POINTER, STATUS_OK};

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
}

