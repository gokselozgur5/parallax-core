//! Parallax Core Rust Engine: Zero-dependency C-ABI interface for high-performance tensor ops.

pub mod tensor;

use std::slice;

/// Return error codes for C-ABI calls.
pub const STATUS_OK: i32 = 0;
pub const STATUS_NULL_POINTER: i32 = -1;
pub const STATUS_INVALID_DIMENSION: i32 = -2;

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
/// Returns:
/// 0 on success, negative status code on error.
#[no_mangle]
pub unsafe extern "C" fn parallax_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_ptr: *const f32,
    b_ptr: *const f32,
    beta: f32,
    c_ptr: *mut f32,
) -> i32 {
    if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
        return STATUS_NULL_POINTER;
    }

    let a = slice::from_raw_parts(a_ptr, m * k);
    let b = slice::from_raw_parts(b_ptr, k * n);
    let c = slice::from_raw_parts_mut(c_ptr, m * n);

    match tensor::gemm_f32(m, n, k, alpha, a, b, beta, c) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_INVALID_DIMENSION,
    }
}

/// C-ABI export for in-place Softmax along last dimension.
///
/// Parameters:
/// - `data_ptr`: Pointer to contiguous array of size `rows * cols` (in/out)
/// - `rows`: Number of rows
/// - `cols`: Number of columns (softmax applied per row)
///
/// Returns:
/// 0 on success, negative status code on error.
#[no_mangle]
pub unsafe extern "C" fn parallax_softmax_f32(
    data_ptr: *mut f32,
    rows: usize,
    cols: usize,
) -> i32 {
    if data_ptr.is_null() {
        return STATUS_NULL_POINTER;
    }

    let data = slice::from_raw_parts_mut(data_ptr, rows * cols);

    match tensor::softmax_2d(data, rows, cols) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_INVALID_DIMENSION,
    }
}

/// C-ABI export for version check / smoke test.
#[no_mangle]
pub extern "C" fn parallax_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c_abi_version() {
        assert_eq!(parallax_version(), 1);
    }

    #[test]
    fn test_c_abi_gemm() {
        let a = vec![2.0f32, 0.0, 0.0, 2.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];

        unsafe {
            let res = parallax_gemm_f32(
                2, 2, 2,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                c.as_mut_ptr(),
            );
            assert_eq!(res, STATUS_OK);
        }

        assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_c_abi_softmax() {
        let mut data = vec![0.0f32, 0.0f32];
        unsafe {
            let res = parallax_softmax_f32(data.as_mut_ptr(), 1, 2);
            assert_eq!(res, STATUS_OK);
        }
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);
    }
}
