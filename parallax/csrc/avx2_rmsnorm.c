#include <stddef.h>
#include <math.h>

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define PARALLAX_NO_AVX_TARGET __attribute__((noinline, target("no-avx,no-fma")))
#else
#define PARALLAX_NO_AVX_TARGET
#endif

PARALLAX_NO_AVX_TARGET
int c_has_avx2_rmsnorm(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((noinline, target("avx2,fma")))
static void rmsnorm_avx2_kernel(const float *x, const float *w, float *o, int r, int c, float eps) {
    const size_t rows = (size_t)r, cols = (size_t)c;
    for (size_t row = 0; row < rows; ++row) {
        const float *xr = x + row * cols;
        float *orow = o + row * cols;
        __m256d sv0 = _mm256_setzero_pd(), sv1 = _mm256_setzero_pd();
        size_t col = 0;
        for (; cols - col >= 8; col += 8) {
            __m256 xv = _mm256_loadu_ps(xr + col);
            __m256d x0 = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
            __m256d x1 = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
            sv0 = _mm256_fmadd_pd(x0, x0, sv0);
            sv1 = _mm256_fmadd_pd(x1, x1, sv1);
        }
        double t0[4], t1[4], sum_sq = 0.0;
        _mm256_storeu_pd(t0, sv0); _mm256_storeu_pd(t1, sv1);
        for (int k = 0; k < 4; ++k) sum_sq += t0[k] + t1[k];
        for (; col < cols; ++col) sum_sq += (double)xr[col] * (double)xr[col];
        float inv_rms = 1.0f / (float)sqrt(sum_sq / (double)cols + (double)eps);
        __m256 inv_vec = _mm256_set1_ps(inv_rms);
        for (col = 0; cols - col >= 8; col += 8) {
            __m256 out_v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + col), inv_vec), _mm256_loadu_ps(w + col));
            _mm256_storeu_ps(orow + col, out_v);
        }
        for (; col < cols; ++col) orow[col] = (xr[col] * inv_rms) * w[col];
    }
}
#endif

PARALLAX_NO_AVX_TARGET
int c_rmsnorm_avx2_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    if (!x || !w || !o) return PARALLAX_STATUS_NULL_POINTER;
    if (r <= 0 || c <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx2_rmsnorm()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    rmsnorm_avx2_kernel(x, w, o, r, c, eps);
    return PARALLAX_STATUS_OK;
#else
    (void)eps;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}
