//! Tensor operations module with pure Rust and zero external dependencies.

/// In-place softmax along the last dimension of a 2D matrix or 1D array.
/// `data`: slice of floats representing rows contiguous in memory.
/// `rows`: number of rows.
/// `cols`: number of columns per row.
pub fn softmax_2d(data: &mut [f32], rows: usize, cols: usize) -> Result<(), &'static str> {
    let req_len = match rows.checked_mul(cols) {
        Some(v) => v,
        None => return Err("rows * cols overflow"),
    };
    if data.len() != req_len {
        return Err("data length does not match rows * cols");
    }
    if cols == 0 {
        return Ok(());
    }

    for r in 0..rows {
        let row_slice = &mut data[r * cols..(r + 1) * cols];

        if row_slice.iter().any(|v| v.is_nan()) {
            for val in row_slice.iter_mut() {
                *val = f32::NAN;
            }
            continue;
        }

        // Numerical stability: find max
        let mut max_val = row_slice[0];
        for &val in row_slice.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }

        if max_val.is_infinite() {
            for val in row_slice.iter_mut() {
                *val = f32::NAN;
            }
            continue;
        }

        // Exponentiate and sum
        let mut sum_exp = 0.0f32;
        for val in row_slice.iter_mut() {
            *val = (*val - max_val).exp();
            sum_exp += *val;
        }

        // Normalize
        let inv_sum = 1.0 / sum_exp;
        for val in row_slice.iter_mut() {
            *val *= inv_sum;
        }
    }

    Ok(())
}

/// General Matrix Multiplication (GEMM): C = alpha * (A @ B) + beta * C
/// A: M x K (row-major)
/// B: K x N (row-major)
/// C: M x N (row-major)
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) -> Result<(), &'static str> {
    let req_a = match m.checked_mul(k) {
        Some(v) => v,
        None => return Err("matrix A dimension overflow"),
    };
    let req_b = match k.checked_mul(n) {
        Some(v) => v,
        None => return Err("matrix B dimension overflow"),
    };
    let req_c = match m.checked_mul(n) {
        Some(v) => v,
        None => return Err("matrix C dimension overflow"),
    };
    if a.len() != req_a {
        return Err("matrix A dimension mismatch");
    }
    if b.len() != req_b {
        return Err("matrix B dimension mismatch");
    }
    if c.len() != req_c {
        return Err("matrix C dimension mismatch");
    }

    // Scale C by beta first
    if beta == 0.0 {
        for val in c.iter_mut() {
            *val = 0.0;
        }
    } else if beta != 1.0 {
        for val in c.iter_mut() {
            *val *= beta;
        }
    }

    if alpha == 0.0 || k == 0 {
        return Ok(());
    }

    // Standard cache-friendly row-major GEMM: i -> p -> j loop order
    for i in 0..m {
        let a_row_offset = i * k;
        let c_row_offset = i * n;
        for p in 0..k {
            let a_val = a[a_row_offset + p] * alpha;
            let b_row_offset = p * n;
            for j in 0..n {
                c[c_row_offset + j] += a_val * b[b_row_offset + j];
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_identity() {
        let m = 2;
        let k = 2;
        let n = 2;

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let mut c = vec![0.0; 4];

        let res = gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c);
        assert!(res.is_ok());
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gemm_general() {
        // [1, 2, 3] x [[7, 8], [9, 1], [2, 3]]
        // A is 1x3, B is 3x2, C is 1x2
        // C[0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // C[1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0];
        let mut c = vec![5.0, 5.0]; // beta = 1.0 -> C = A@B + C

        let res = gemm_f32(1, 2, 3, 1.0, &a, &b, 1.0, &mut c);
        assert!(res.is_ok());
        assert_eq!(c, vec![36.0, 24.0]);
    }

    #[test]
    fn test_softmax() {
        let mut data = vec![0.0, 0.0, 1.0, 2.0]; // 2 rows, 2 cols
        let res = softmax_2d(&mut data, 2, 2);
        assert!(res.is_ok());

        // Row 0: [0.0, 0.0] -> [0.5, 0.5]
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 0.5).abs() < 1e-6);

        // Row 1: [1.0, 2.0] -> sum = e^-1 + 1 = 0.367879 + 1 = 1.367879
        // data[2] = 0.367879 / 1.367879 ~= 0.2689414
        // data[3] = 1.0 / 1.367879 ~= 0.7310586
        assert!((data[2] - 0.26894142).abs() < 1e-5);
        assert!((data[3] - 0.7310586).abs() < 1e-5);
        assert!(((data[2] + data[3]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_nan_and_infinities() {
        let mut data = vec![1.0, f32::NAN, 3.0, 1.0, f32::NEG_INFINITY, 2.0];
        let res = softmax_2d(&mut data, 2, 3);
        assert!(res.is_ok());
        assert!(data[0].is_nan() && data[1].is_nan() && data[2].is_nan());
        assert!((data[3] - 0.26894143).abs() < 1e-5);
        assert_eq!(data[4], 0.0);
        assert!((data[5] - 0.7310586).abs() < 1e-5);
    }
}
