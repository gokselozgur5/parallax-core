//! Parallax Core Rust Engine: Zero-dependency C-ABI interface for high-performance tensor ops.

pub mod bpe;
pub mod c_abi;
pub mod flash_attn;
pub mod kv_cache;
pub mod neon_fp16;
pub mod scheduler;
pub mod speculative;
pub mod storage;
pub mod tensor;

use std::slice;
/// Return error codes for C-ABI calls.
pub const STATUS_OK: i32 = 0;
pub const STATUS_NULL_POINTER: i32 = -1;
pub const STATUS_INVALID_DIMENSION: i32 = -2;
pub const STATUS_UNALIGNED_POINTER: i32 = -3;
pub const STATUS_ALIASING_POINTER: i32 = -4;
pub const STATUS_NOT_FOUND: i32 = -5;
#[inline]
fn is_aligned_u32<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(std::mem::align_of::<u32>())
}

#[inline]
fn is_aligned_f32<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(std::mem::align_of::<f32>())
}
#[inline]
fn is_aligned_usize<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(std::mem::align_of::<usize>())
}

#[inline]
fn is_aligned_ptr<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(std::mem::align_of::<*const ()>())
}

#[inline]
fn ranges_overlap_u32(p1: usize, len1: usize, p2: usize, len2: usize) -> bool {
    if len1 == 0 || len2 == 0 {
        return false;
    }
    let Some(size1) = len1.checked_mul(std::mem::size_of::<u32>()) else {
        return true;
    };
    let Some(size2) = len2.checked_mul(std::mem::size_of::<u32>()) else {
        return true;
    };
    let Some(end1) = p1.checked_add(size1) else {
        return true;
    };
    let Some(end2) = p2.checked_add(size2) else {
        return true;
    };
    p1 < end2 && p2 < end1
}

#[inline]
fn ranges_overlap(p1: usize, len1: usize, p2: usize, len2: usize) -> bool {
    if len1 == 0 || len2 == 0 {
        return false;
    }
    let Some(size1) = len1.checked_mul(std::mem::size_of::<f32>()) else {
        return true;
    };
    let Some(size2) = len2.checked_mul(std::mem::size_of::<f32>()) else {
        return true;
    };
    let Some(end1) = p1.checked_add(size1) else {
        return true;
    };
    let Some(end2) = p2.checked_add(size2) else {
        return true;
    };
    p1 < end2 && p2 < end1
}
/// C-ABI export for GEMM: C = alpha * (A @ B) + beta * C
///
/// Parameters:
/// - `m`: Number of rows in A and C
/// - `n`: Number of cols in B and C
/// - `k`: Number of cols in A and rows in B
/// - `alpha`: Scalar multiplier for A @ B
/// - `a_ptr`: Pointer to row-major array of size m * k
/// - `b_ptr`: Pointer to row-major array of size k * n
/// - `beta`: Scalar multiplier for C
/// - `c_ptr`: Pointer to row-major array of size m * n (in/out)
///
/// # Safety
///
/// Pointers `a_ptr`, `b_ptr`, and `c_ptr` must be valid, 4-byte aligned, non-null,
/// and reference at least `a_len`, `b_len`, and `c_len` elements respectively.
///
/// Returns:
/// 0 on success, negative status code on error.
#[no_mangle]
pub unsafe extern "C" fn parallax_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_len: usize,
    b_ptr: *const f32,
    b_len: usize,
    beta: f32,
    c_ptr: *mut f32,
    c_len: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_f32(a_ptr) || !is_aligned_f32(b_ptr) || !is_aligned_f32(c_ptr) {
            return STATUS_UNALIGNED_POINTER;
        }
        let (ra, rb, rc) = (m.checked_mul(k), k.checked_mul(n), m.checked_mul(n));
        let (Some(ra), Some(rb), Some(rc)) = (ra, rb, rc) else {
            return STATUS_INVALID_DIMENSION;
        };
        if a_len != ra || b_len != rb || c_len != rc {
            return STATUS_INVALID_DIMENSION;
        }
        if ranges_overlap(c_ptr as usize, c_len, a_ptr as usize, a_len)
            || ranges_overlap(c_ptr as usize, c_len, b_ptr as usize, b_len)
        {
            return STATUS_ALIASING_POINTER;
        }
        let (a, b) = (
            slice::from_raw_parts(a_ptr, a_len),
            slice::from_raw_parts(b_ptr, b_len),
        );
        let c = slice::from_raw_parts_mut(c_ptr, c_len);
        match tensor::gemm_f32(m, n, k, alpha, a, b, beta, c) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_DIMENSION,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}
/// C-ABI export for in-place Softmax along last dimension.
///
/// Parameters:
/// - `data_ptr`: Pointer to contiguous array of size `rows * cols` (in/out)
/// - `rows`: Number of rows
/// - `cols`: Number of columns (softmax applied per row)
///
/// # Safety
///
/// `data_ptr` must be valid, 4-byte aligned, non-null, and reference at least
/// `data_len` elements.
///
/// Returns:
/// 0 on success, negative status code on error.
#[no_mangle]
pub unsafe extern "C" fn parallax_softmax_f32(
    data_ptr: *mut f32,
    data_len: usize,
    rows: usize,
    cols: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if data_ptr.is_null() {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_f32(data_ptr) {
            return STATUS_UNALIGNED_POINTER;
        }
        let Some(req) = rows.checked_mul(cols) else {
            return STATUS_INVALID_DIMENSION;
        };
        if data_len != req {
            return STATUS_INVALID_DIMENSION;
        }
        let data = slice::from_raw_parts_mut(data_ptr, data_len);
        match tensor::softmax_2d(data, rows, cols) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_DIMENSION,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}
/// C-ABI export for looking up a single BPE merge rule via binary search.
///
/// # Safety
///
/// Pointers `table_ptr`, `out_rank`, and `out_new_id` must be valid, aligned,
/// non-null, and `table_ptr` must reference at least `table_len` elements.
#[no_mangle]
pub unsafe extern "C" fn parallax_bpe_find_merge(
    table_ptr: *const bpe::MergeEntry,
    table_len: usize,
    left: u32,
    right: u32,
    out_rank: *mut u32,
    out_new_id: *mut u32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if table_ptr.is_null() || out_rank.is_null() || out_new_id.is_null() {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_u32(table_ptr) || !is_aligned_u32(out_rank) || !is_aligned_u32(out_new_id) {
            return STATUS_UNALIGNED_POINTER;
        }
        let table = slice::from_raw_parts(table_ptr, table_len);
        match bpe::find_merge(table, left, right) {
            Some(entry) => {
                *out_rank = entry.rank;
                *out_new_id = entry.new_id;
                STATUS_OK
            }
            None => STATUS_NOT_FOUND,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}
/// C-ABI export for validating whether a merge table is strictly sorted.
///
/// # Safety
///
/// If `table_len > 0`, `table_ptr` must be valid, aligned, non-null,
/// and reference at least `table_len` elements.
#[no_mangle]
pub unsafe extern "C" fn parallax_bpe_is_sorted(
    table_ptr: *const bpe::MergeEntry,
    table_len: usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if table_ptr.is_null() && table_len > 0 {
            return STATUS_NULL_POINTER;
        }
        if table_len > 0 && !is_aligned_u32(table_ptr) {
            return STATUS_UNALIGNED_POINTER;
        }
        let table = if table_len == 0 {
            &[][..]
        } else {
            slice::from_raw_parts(table_ptr, table_len)
        };
        if bpe::is_table_sorted(table) {
            1
        } else {
            0
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}

#[inline]
unsafe fn validate_flash_attn_ptrs(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
) -> i32 {
    if q.is_null() || k.is_null() || v.is_null() || out.is_null() {
        return STATUS_NULL_POINTER;
    }
    if !is_aligned_f32(q) || !is_aligned_f32(k) || !is_aligned_f32(v) || !is_aligned_f32(out) {
        return STATUS_UNALIGNED_POINTER;
    }
    STATUS_OK
}
/// C-ABI export for single-head flash attention f32.
///
/// # Safety
///
/// Pointers `q`, `k`, `v`, and `out` must be valid, 4-byte aligned, non-null,
/// and reference at least `seq_len * head_dim` elements.
#[no_mangle]
pub unsafe extern "C" fn parallax_flash_attn_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    is_causal: i32,
    f16_corr: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let p_status = validate_flash_attn_ptrs(q, k, v, out);
        if p_status != STATUS_OK {
            return p_status;
        }
        let Some(len) = seq_len.checked_mul(head_dim) else {
            return STATUS_INVALID_DIMENSION;
        };
        if len > (isize::MAX as usize) / std::mem::size_of::<f32>() {
            return STATUS_INVALID_DIMENSION;
        }
        if ranges_overlap(out as usize, len, q as usize, len)
            || ranges_overlap(out as usize, len, k as usize, len)
            || ranges_overlap(out as usize, len, v as usize, len)
        {
            return STATUS_ALIASING_POINTER;
        }
        let (qs, ks, vs) = (
            slice::from_raw_parts(q, len),
            slice::from_raw_parts(k, len),
            slice::from_raw_parts(v, len),
        );
        let outs = slice::from_raw_parts_mut(out, len);
        let cfg = flash_attn::FlashAttnConfig {
            scale,
            is_causal: is_causal != 0,
            f16_scale_correction: f16_corr != 0,
        };
        match flash_attn::flash_attn(qs, ks, vs, outs, seq_len, head_dim, cfg) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_DIMENSION,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}
/// C-ABI export for version check / smoke test.
#[no_mangle]
pub extern "C" fn parallax_version() -> u32 {
    3
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_c_abi_version() {
        assert_eq!(parallax_version(), 3);
    }
    #[test]
    fn test_c_abi_gemm() {
        let a = [2.0f32, 0.0, 0.0, 2.0];
        let b = [1.0f32, 2.0, 3.0, 4.0];
        let mut c = [0.0f32; 4];
        unsafe {
            let res = parallax_gemm_f32(
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                a.len(),
                b.as_ptr(),
                b.len(),
                0.0,
                c.as_mut_ptr(),
                c.len(),
            );
            assert_eq!(res, STATUS_OK);
        }
        assert_eq!(&c[..], &[2.0, 4.0, 6.0, 8.0]);
    }
    #[test]
    fn test_c_abi_softmax() {
        let mut data = [0.0f32, 0.0f32];
        unsafe {
            let res = parallax_softmax_f32(data.as_mut_ptr(), data.len(), 1, 2);
            assert_eq!(res, STATUS_OK);
        }
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_c_abi_validation() {
        let buf = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        unsafe {
            let res = parallax_gemm_f32(
                2,
                2,
                2,
                1.0,
                buf.as_ptr(),
                2,
                buf.as_ptr(),
                4,
                0.0,
                out.as_mut_ptr(),
                4,
            );
            assert_eq!(res, STATUS_INVALID_DIMENSION);
            let res = parallax_gemm_f32(
                usize::MAX,
                2,
                2,
                1.0,
                buf.as_ptr(),
                4,
                buf.as_ptr(),
                4,
                0.0,
                out.as_mut_ptr(),
                4,
            );
            assert_eq!(res, STATUS_INVALID_DIMENSION);
            let res = parallax_softmax_f32(out.as_mut_ptr(), 2, 2, 2);
            assert_eq!(res, STATUS_INVALID_DIMENSION);
            // Null and alignment checks
            assert_eq!(
                parallax_gemm_f32(
                    2,
                    2,
                    2,
                    1.0,
                    std::ptr::null(),
                    4,
                    buf.as_ptr(),
                    4,
                    0.0,
                    out.as_mut_ptr(),
                    4
                ),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                parallax_softmax_f32(std::ptr::null_mut(), 4, 2, 2),
                STATUS_NULL_POINTER
            );
            let raw: [u8; 16] = [0; 16];
            let unaligned = raw.as_ptr().wrapping_add(1) as *const f32;
            let unaligned_mut = raw.as_ptr().wrapping_add(1) as *mut f32;
            assert_eq!(
                parallax_gemm_f32(
                    2,
                    2,
                    2,
                    1.0,
                    unaligned,
                    4,
                    buf.as_ptr(),
                    4,
                    0.0,
                    out.as_mut_ptr(),
                    4
                ),
                STATUS_UNALIGNED_POINTER
            );
            assert_eq!(
                parallax_softmax_f32(unaligned_mut, 4, 2, 2),
                STATUS_UNALIGNED_POINTER
            );
            // BPE C-ABI checks
            let table = [bpe::MergeEntry::new(1, 2, 0, 10)];
            let mut rank = 0u32;
            let mut new_id = 0u32;
            let res = parallax_bpe_find_merge(table.as_ptr(), 1, 1, 2, &mut rank, &mut new_id);
            assert_eq!(res, STATUS_OK);
            assert_eq!(rank, 0);
            assert_eq!(new_id, 10);
            let res_missing =
                parallax_bpe_find_merge(table.as_ptr(), 1, 9, 9, &mut rank, &mut new_id);
            assert_eq!(res_missing, STATUS_NOT_FOUND);
            assert_eq!(parallax_bpe_is_sorted(table.as_ptr(), 1), 1);
            assert_eq!(
                parallax_bpe_find_merge(std::ptr::null(), 1, 1, 2, &mut rank, &mut new_id),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                parallax_bpe_is_sorted(std::ptr::null(), 1),
                STATUS_NULL_POINTER
            );
            // Aliasing checks
            let mut shared = vec![1.0f32; 8];
            let s_ptr = shared.as_mut_ptr();
            assert_eq!(
                parallax_gemm_f32(2, 2, 2, 1.0, s_ptr, 4, buf.as_ptr(), 4, 0.0, s_ptr, 4),
                STATUS_ALIASING_POINTER
            );
            assert_eq!(
                parallax_gemm_f32(
                    2,
                    2,
                    2,
                    1.0,
                    s_ptr,
                    4,
                    buf.as_ptr(),
                    4,
                    0.0,
                    s_ptr.add(2),
                    4
                ),
                STATUS_ALIASING_POINTER
            );
        }
    }

    #[test]
    fn test_c_abi_flash_attn() {
        let q = [1.0f32, 2.0];
        let k = [1.0f32, 2.0];
        let v = [3.0f32, 4.0];
        let mut out = [0.0f32; 2];
        unsafe {
            let res = parallax_flash_attn_f32(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                1,
                2,
                0.5,
                0,
                1,
            );
            assert_eq!(res, STATUS_OK);
            assert!((out[0] - 3.0).abs() < 1e-4);
            assert!((out[1] - 4.0).abs() < 1e-4);
            let res_overlap = parallax_flash_attn_f32(
                out.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
                1,
                2,
                0.5,
                0,
                1,
            );
            assert_eq!(res_overlap, STATUS_ALIASING_POINTER);
        }
    }
}

#[cfg(test)]
#[test]
fn test_flash_attn_f32_rejects_slice_over_isize_max() {
    let q = [1.0f32; 2];
    let k = [1.0f32; 2];
    let v = [1.0f32; 2];
    let mut out = [0.0f32; 2];
    unsafe {
        let status_u64_wrap = parallax_flash_attn_f32(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            1usize << 33,
            1usize << 29,
            0.5,
            0,
            0,
        );
        assert_eq!(status_u64_wrap, STATUS_INVALID_DIMENSION);
        let status_isize_edge = parallax_flash_attn_f32(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            1usize << 32,
            1usize << 29,
            0.5,
            0,
            0,
        );
        assert_eq!(status_isize_edge, STATUS_INVALID_DIMENSION);
    }
}

/// C-ABI export to find bucket boundary for given sequence length.
///
/// # Safety
///
/// `buckets_ptr` must reference at least `buckets_len` valid `usize` values.
/// `out_bucket` must be a valid, aligned, non-null pointer to a `usize`.
#[no_mangle]
pub unsafe extern "C" fn parallax_find_bucket(
    seq_len: usize,
    buckets_ptr: *const usize,
    buckets_len: usize,
    out_bucket: *mut usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if buckets_ptr.is_null() || out_bucket.is_null() {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_usize(buckets_ptr) || !is_aligned_usize(out_bucket) {
            return STATUS_UNALIGNED_POINTER;
        }
        if buckets_len == 0 {
            return STATUS_INVALID_DIMENSION;
        }
        let buckets = slice::from_raw_parts(buckets_ptr, buckets_len);
        let binner = match scheduler::SequenceBinner::new(buckets) {
            Ok(b) => b,
            Err(_) => return STATUS_INVALID_DIMENSION,
        };
        match binner.find_bucket(seq_len) {
            Ok(b) => {
                *out_bucket = b;
                STATUS_OK
            }
            Err(_) => STATUS_INVALID_DIMENSION,
        }
    }))
    .unwrap_or(STATUS_INVALID_DIMENSION)
}

/// C-ABI export to collate sequences into padded buffer.
///
/// # Safety
///
/// All sequence, bucket, and output pointers must be valid, non-null, aligned,
/// and reference slices matching their respective length parameters.
#[no_mangle]
pub unsafe extern "C" fn parallax_collate_batch_u32(
    seq_ptrs: *const *const u32,
    seq_lens: *const usize,
    num_seqs: usize,
    buckets_ptr: *const usize,
    buckets_len: usize,
    pad_token: u32,
    out_tokens: *mut u32,
    out_mask: *mut u32,
    out_len: usize,
    out_bucket_len: *mut usize,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if seq_ptrs.is_null()
            || seq_lens.is_null()
            || buckets_ptr.is_null()
            || out_tokens.is_null()
            || out_mask.is_null()
            || out_bucket_len.is_null()
        {
            return STATUS_NULL_POINTER;
        }
        if !is_aligned_ptr(seq_ptrs)
            || !is_aligned_usize(seq_lens)
            || !is_aligned_usize(buckets_ptr)
            || !is_aligned_u32(out_tokens)
            || !is_aligned_u32(out_mask)
            || !is_aligned_usize(out_bucket_len)
        {
            return STATUS_UNALIGNED_POINTER;
        }
        if ranges_overlap_u32(out_tokens as usize, out_len, out_mask as usize, out_len) {
            return STATUS_ALIASING_POINTER;
        }
        let buckets = slice::from_raw_parts(buckets_ptr, buckets_len);
        let Ok(binner) = scheduler::SequenceBinner::new(buckets) else {
            return STATUS_INVALID_DIMENSION;
        };
        let lens = slice::from_raw_parts(seq_lens, num_seqs);
        let ptrs = slice::from_raw_parts(seq_ptrs, num_seqs);
        let mut seqs = Vec::with_capacity(num_seqs);
        for i in 0..num_seqs {
            if ptrs[i].is_null() { return STATUS_NULL_POINTER; }
            if !is_aligned_u32(ptrs[i]) { return STATUS_UNALIGNED_POINTER; }
            seqs.push(slice::from_raw_parts(ptrs[i], lens[i]));
        }
        let toks = slice::from_raw_parts_mut(out_tokens, out_len);
        let mask = slice::from_raw_parts_mut(out_mask, out_len);
        match binner.collate(&seqs, pad_token, toks, mask) {
            Ok(b) => {
                *out_bucket_len = b;
                STATUS_OK
            }
            Err(_) => STATUS_INVALID_DIMENSION,
        }
    })).unwrap_or(STATUS_INVALID_DIMENSION)
}


pub mod q8_gemv;
pub use q8_gemv::{gemv_q8_0, vec_dot_q8_0_f32, vec_dot_q8_0_f32_block, BlockQ8_0};
