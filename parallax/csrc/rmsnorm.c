#include <stddef.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#endif

/*
 * RMSNorm forward pass with SIMD acceleration (float32):
 * output[r, c] = (x[r, c] / rms) * weight[c]
 * where rms = sqrt(mean(x[r, :]^2) + eps)
 */
void c_rmsnorm_f32(const float *x, const float *weight, float *output, int rows, int cols, float eps) {
    if (!x || !weight || !output || rows <= 0 || cols <= 0) return;

    const size_t rows_s = (size_t)rows;
    const size_t cols_s = (size_t)cols;

    for (size_t r = 0; r < rows_s; ++r) {
        const float *x_row = x + r * cols_s;
        float *out_row = output + r * cols_s;

        // Step 1: Calculate sum of squares
        double sum_sq = 0.0;
        size_t c = 0;

#if defined(USE_NEON)
        float64x2_t sv0 = vdupq_n_f64(0.0), sv1 = vdupq_n_f64(0.0);
        for (; cols_s - c >= 4; c += 4) {
            float32x4_t xv = vld1q_f32(x_row + c);
            float64x2_t x0 = vcvt_f64_f32(vget_low_f32(xv));
            float64x2_t x1 = vcvt_f64_f32(vget_high_f32(xv));
            sv0 = vmlaq_f64(sv0, x0, x0);
            sv1 = vmlaq_f64(sv1, x1, x1);
        }
        double t0[2], t1[2];
        vst1q_f64(t0, sv0); vst1q_f64(t1, sv1);
        sum_sq = (t0[0] + t0[1]) + (t1[0] + t1[1]);
#elif defined(USE_AVX2)
        __m256d sv0 = _mm256_setzero_pd(), sv1 = _mm256_setzero_pd();
        for (; cols_s - c >= 8; c += 8) {
            __m256 xv = _mm256_loadu_ps(x_row + c);
            __m256d x0 = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
            __m256d x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
            sv0 = _mm256_fmadd_pd(x0, x0, sv0);
            sv1 = _mm256_fmadd_pd(x1, x1, sv1);
        }
        double t0[4], t1[4];
        _mm256_storeu_pd(t0, sv0); _mm256_storeu_pd(t1, sv1);
        for (int k = 0; k < 4; ++k) sum_sq += t0[k] + t1[k];
#endif

        for (; c < cols_s; ++c) {
            double v = (double)x_row[c];
            sum_sq += v * v;
        }

        // Step 2: Compute RMS and normalization factor
        double mean_sq = sum_sq * (1.0 / (double)cols_s);
        float rms = (float)sqrt(mean_sq + (double)eps);
        float inv_rms = 1.0f / rms;

        // Step 3: Compute output = (x / rms) * weight
        c = 0;
#if defined(USE_NEON)
        float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
        for (; cols_s - c >= 4; c += 4) {
            float32x4_t xv = vld1q_f32(x_row + c);
            float32x4_t wv = vld1q_f32(weight + c);
            float32x4_t normed = vmulq_f32(xv, inv_rms_vec);
            float32x4_t out_v = vmulq_f32(normed, wv);
            vst1q_f32(out_row + c, out_v);
        }
#elif defined(USE_AVX2)
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
        for (; cols_s - c >= 8; c += 8) {
            __m256 xv = _mm256_loadu_ps(x_row + c);
            __m256 wv = _mm256_loadu_ps(weight + c);
            __m256 normed = _mm256_mul_ps(xv, inv_rms_vec);
            __m256 out_v = _mm256_mul_ps(normed, wv);
            _mm256_storeu_ps(out_row + c, out_v);
        }
#endif

        for (; c < cols_s; ++c) {
            out_row[c] = (x_row[c] * inv_rms) * weight[c];
        }
    }
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

#if defined(USE_NEON)
        float64x2_t sum_vec = vdupq_n_f64(0.0);
        for (; cols_s - c >= 2; c += 2) {
            float64x2_t xv = vld1q_f64(x_row + c);
            sum_vec = vmlaq_f64(sum_vec, xv, xv);
        }
        double tmp[2];
        vst1q_f64(tmp, sum_vec);
        sum_sq = tmp[0] + tmp[1];
#elif defined(USE_AVX2)
        __m256d sum_vec = _mm256_setzero_pd();
        for (; cols_s - c >= 4; c += 4) {
            __m256d xv = _mm256_loadu_pd(x_row + c);
            sum_vec = _mm256_fmadd_pd(xv, xv, sum_vec);
        }
        double tmp[4];
        _mm256_storeu_pd(tmp, sum_vec);
        for (int k = 0; k < 4; ++k) sum_sq += tmp[k];
#endif

        for (; c < cols_s; ++c) {
            double v = x_row[c];
            sum_sq += v * v;
        }

        double mean_sq = sum_sq * inv_cols;
        double rms = sqrt(mean_sq + eps);
        double inv_rms = 1.0 / rms;

        c = 0;
#if defined(USE_NEON)
        float64x2_t inv_rms_vec = vdupq_n_f64(inv_rms);
        for (; cols_s - c >= 2; c += 2) {
            float64x2_t xv = vld1q_f64(x_row + c);
            float64x2_t wv = vld1q_f64(weight + c);
            float64x2_t normed = vmulq_f64(xv, inv_rms_vec);
            float64x2_t out_v = vmulq_f64(normed, wv);
            vst1q_f64(out_row + c, out_v);
        }
#elif defined(USE_AVX2)
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
