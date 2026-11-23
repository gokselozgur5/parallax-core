#include "parallax_simd.h"
#include <math.h>

PARALLAX_SCALAR_TARGET
void c_rmsnorm_scalar_f32(const float *x, const float *weight, float *output, int rows, int cols, float eps) {
    if (!x || !weight || !output || rows <= 0 || cols <= 0) return;
    const size_t rows_s = (size_t)rows, cols_s = (size_t)cols;
    for (size_t r = 0; r < rows_s; ++r) {
        const float *x_row = x + r * cols_s;
        float *out_row = output + r * cols_s;
        double sum_sq = 0.0;
        for (size_t c = 0; c < cols_s; ++c) {
            double v = (double)x_row[c];
            sum_sq += v * v;
        }
        float inv_rms = 1.0f / (float)sqrt(sum_sq / (double)cols_s + (double)eps);
        for (size_t c = 0; c < cols_s; ++c) {
            out_row[c] = (x_row[c] * inv_rms) * weight[c];
        }
    }
}

void c_rmsnorm_f32(const float *x, const float *weight, float *output, int rows, int cols, float eps) {
    if (!x || !weight || !output || rows <= 0 || cols <= 0) return;
    if (c_has_avx512_rmsnorm_f32() && c_rmsnorm_avx512_f32(x, weight, output, rows, cols, eps) == PARALLAX_STATUS_OK) return;
    if (c_has_avx2_rmsnorm() && c_rmsnorm_avx2_f32(x, weight, output, rows, cols, eps) == PARALLAX_STATUS_OK) return;
    if (c_has_neon_rmsnorm()) {
        c_rmsnorm_neon_f32(x, weight, output, rows, cols, eps);
        return;
    }
    c_rmsnorm_scalar_f32(x, weight, output, rows, cols, eps);
}

/*
 * RMSNorm forward pass with SIMD acceleration (float64):
 * output[r, c] = (x[r, c] / rms) * weight[c]
 * where rms = sqrt(mean(x[r, :]^2) + eps)
 */
void c_rmsnorm_f64(const double *x, const double *weight, double *output, int rows, int cols, double eps) {
    if (!x || !weight || !output || rows <= 0 || cols <= 0) return;

    const double inv_cols = 1.0 / (double)cols;
    const size_t rows_s = (size_t)rows;
    const size_t cols_s = (size_t)cols;

    for (size_t r = 0; r < rows_s; ++r) {
        const double *x_row = x + r * cols_s;
        double *out_row = output + r * cols_s;

        double sum_sq = 0.0;
        size_t c = 0;

#if defined(PARALLAX_HAVE_NEON)
        float64x2_t sum_vec = vdupq_n_f64(0.0);
        for (; cols_s - c >= 2; c += 2) {
            float64x2_t xv = vld1q_f64(x_row + c);
            sum_vec = vmlaq_f64(sum_vec, xv, xv);
        }
        double tmp[2];
        vst1q_f64(tmp, sum_vec);
        sum_sq = tmp[0] + tmp[1];
#elif defined(PARALLAX_HAVE_AVX2)
        __m256d sum_vec = _mm256_setzero_pd();
        for (; cols_s - c >= 4; c += 4) {
            __m256d xv = _mm256_loadu_pd(x_row + c);
            sum_vec = _mm256_fmadd_pd(xv, xv, sum_vec);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, sum_vec);
        for (size_t k = 0; k < 4; ++k) sum_sq += tmp[k];
#endif

        for (; c < cols_s; ++c) {
            double v = x_row[c];
            sum_sq += v * v;
        }

        double mean_sq = sum_sq * inv_cols;
        double rms = sqrt(mean_sq + eps);
        double inv_rms = 1.0 / rms;

        c = 0;
#if defined(PARALLAX_HAVE_NEON)
        float64x2_t inv_rms_vec = vdupq_n_f64(inv_rms);
        for (; cols_s - c >= 2; c += 2) {
            float64x2_t xv = vld1q_f64(x_row + c);
            float64x2_t wv = vld1q_f64(weight + c);
            float64x2_t normed = vmulq_f64(xv, inv_rms_vec);
            float64x2_t out_v = vmulq_f64(normed, wv);
            vst1q_f64(out_row + c, out_v);
        }
#elif defined(PARALLAX_HAVE_AVX2)
        __m256d inv_rms_vec = _mm256_set1_pd(inv_rms);
        for (; cols_s - c >= 4; c += 4) {
            __m256d xv = _mm256_loadu_pd(x_row + c);
            __m256d wv = _mm256_loadu_pd(weight + c);
            __m256d normed = _mm256_mul_pd(xv, inv_rms_vec);
            __m256d out_v = _mm256_mul_pd(normed, wv);
            _mm256_storeu_pd(out_row + c, out_v);
        }
#endif

        for (; c < cols_s; ++c) {
            out_row[c] = (x_row[c] * inv_rms) * weight[c];
        }
    }
}
