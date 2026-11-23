use parallax_rs::c_abi::backend::{parallax_get_device_type, DEVICE_TYPE_CPU};
use parallax_rs::c_abi::quant::{parallax_dequantize_q8_0, parallax_quantize_q8_0};
use parallax_rs::{
    parallax_collate_batch_u32, parallax_version, BlockQ8_0, STATUS_ALIASING_POINTER,
    STATUS_NULL_POINTER, STATUS_OK,
};
use std::mem::{align_of, offset_of, size_of};

#[test]
fn test_block_q8_0_layout_and_offsets() {
    assert_eq!(size_of::<BlockQ8_0>(), 36);
    assert_eq!(align_of::<BlockQ8_0>(), 4);
    assert_eq!(offset_of!(BlockQ8_0, scale), 0);
    assert_eq!(offset_of!(BlockQ8_0, qs), 4);
}

#[test]
fn test_parallax_version_bumped() {
    assert_eq!(parallax_version(), 3);
}

#[test]
fn test_parallax_collate_batch_u32_reports_bucket_len() {
    let seq1 = [1u32, 2, 3];
    let seq2 = [4u32, 5];
    let ptrs = [seq1.as_ptr(), seq2.as_ptr()];
    let lens = [seq1.len(), seq2.len()];
    let buckets = [4usize, 8, 16];
    let mut out_tokens = [0u32; 8];
    let mut out_mask = [0u32; 8];
    let mut bucket_len = 0usize;

    unsafe {
        let status = parallax_collate_batch_u32(
            ptrs.as_ptr(),
            lens.as_ptr(),
            2,
            buckets.as_ptr(),
            buckets.len(),
            0,
            out_tokens.as_mut_ptr(),
            out_mask.as_mut_ptr(),
            out_tokens.len(),
            &mut bucket_len,
        );
        assert_eq!(status, STATUS_OK);
    }
    assert_eq!(bucket_len, 4);
    assert_eq!(&out_tokens[..4], &[1, 2, 3, 0]);
    assert_eq!(&out_tokens[4..8], &[4, 5, 0, 0]);
    assert_eq!(&out_mask[..4], &[1, 1, 1, 0]);
    assert_eq!(&out_mask[4..8], &[1, 1, 0, 0]);
}

#[test]
fn test_parallax_collate_batch_u32_validations() {
    let seq1 = [1u32, 2];
    let ptrs = [seq1.as_ptr()];
    let lens = [seq1.len()];
    let buckets = [4usize, 8];
    let mut out_tokens = [0u32; 4];
    let mut out_mask = [0u32; 4];
    let mut bucket_len = 0usize;

    unsafe {
        let null_status = parallax_collate_batch_u32(
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            buckets.as_ptr(),
            buckets.len(),
            0,
            out_tokens.as_mut_ptr(),
            out_mask.as_mut_ptr(),
            4,
            std::ptr::null_mut(),
        );
        assert_eq!(null_status, STATUS_NULL_POINTER);

        let overlap_status = parallax_collate_batch_u32(
            ptrs.as_ptr(),
            lens.as_ptr(),
            1,
            buckets.as_ptr(),
            buckets.len(),
            0,
            out_tokens.as_mut_ptr(),
            out_tokens.as_mut_ptr(),
            4,
            &mut bucket_len,
        );
        assert_eq!(overlap_status, STATUS_ALIASING_POINTER);
    }
}

#[test]
fn test_parallax_quant_and_backend_null_alignment() {
    let mut out_dev = 0u32;
    let mut block = BlockQ8_0::new(0.0, [0i8; 32]);
    let mut out_f32 = [0.0f32; 32];
    let input = [1.0f32; 32];

    unsafe {
        assert_eq!(
            parallax_get_device_type(0, std::ptr::null_mut()),
            STATUS_NULL_POINTER
        );
        assert_eq!(parallax_get_device_type(0, &mut out_dev), STATUS_OK);
        assert_eq!(out_dev, DEVICE_TYPE_CPU);

        assert_eq!(
            parallax_quantize_q8_0(std::ptr::null(), &mut block),
            STATUS_NULL_POINTER
        );
        assert_eq!(
            parallax_quantize_q8_0(input.as_ptr(), &mut block),
            STATUS_OK
        );

        assert_eq!(
            parallax_dequantize_q8_0(std::ptr::null(), out_f32.as_mut_ptr()),
            STATUS_NULL_POINTER
        );
        assert_eq!(
            parallax_dequantize_q8_0(&block, out_f32.as_mut_ptr()),
            STATUS_OK
        );
    }
}
