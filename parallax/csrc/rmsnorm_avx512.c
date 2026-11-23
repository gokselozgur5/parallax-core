#include "parallax_simd.h"
#include <math.h>

PARALLAX_SCALAR_TARGET
int c_has_avx512_rmsnorm_f32(void) {
    return PARALLAX_HAS_AVX512_ISA();
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

__attribute__((noinline, target("avx512f,avx512dq,avx512vl,avx512bw")))
static void rmsnorm_avx512_kernel(const float *x, const float *w, float *out, int rows, int cols, float eps) {
    const size_t rows_s = (size_t)rows, cols_s = (size_t)cols;
    for (size_t r = 0; r < rows_s; ++r) {
        const float *xr = x + r * cols_s;
        float *orow = out + r * cols_s;
        __m512d sv0 = _mm512_setzero_pd(), sv1 = _mm512_setzero_pd();
        size_t c = 0;
        for (; cols_s - c >= 16; c += 16) {
            __m512d x0 = _mm512_cvtps_pd(_mm256_loadu_ps(xr + c));
            __m512d x1 = _mm512_cvtps_pd(_mm256_loadu_ps(xr + c + 8));
            sv0 = _mm512_fmadd_pd(x0, x0, sv0);
            sv1 = _mm512_fmadd_pd(x1, x1, sv1);
        }
        double tmp[8], ssq = 0.0;
        _mm512_storeu_pd(tmp, _mm512_add_pd(sv0, sv1));
        for (size_t k = 0; k < 8; ++k) ssq += tmp[k];
        for (; c < cols_s; ++c) ssq += (double)xr[c] * (double)xr[c];
        float rms = 1.0f / (float)sqrt(ssq / (double)cols_s + (double)eps);
        __m512 rms_vec = _mm512_set1_ps(rms);
        for (c = 0; cols_s - c >= 16; c += 16) {
            __m512 xv = _mm512_loadu_ps(xr + c);
            __m512 wv = _mm512_loadu_ps(w + c);
            _mm512_storeu_ps(orow + c, _mm512_mul_ps(_mm512_mul_ps(xv, rms_vec), wv));
        }
        for (; c < cols_s; ++c) orow[c] = (xr[c] * rms) * w[c];
    }
}
#endif

PARALLAX_SCALAR_TARGET
static void rmsnorm_scalar_fallback(const float *x, const float *w, float *out, int rows, int cols, float eps) {
    const size_t cols_s = (size_t)cols;
    for (size_t r = 0; r < (size_t)rows; ++r) {
        const float *xr = x + r * cols_s;
        float *orow = out + r * cols_s;
        double ssq = 0.0;
        for (size_t c = 0; c < cols_s; ++c) ssq += (double)xr[c] * (double)xr[c];
        float rms = 1.0f / (float)sqrt(ssq / (double)cols_s + (double)eps);
        for (size_t c = 0; c < cols_s; ++c) orow[c] = (xr[c] * rms) * w[c];
    }
}

PARALLAX_SCALAR_TARGET
int c_rmsnorm_avx512_f32(const float *x, const float *w, float *out, int rows, int cols, float eps) {
    if (!x || !w || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (rows <= 0 || cols <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (c_has_avx512_rmsnorm_f32()) {
        rmsnorm_avx512_kernel(x, w, out, rows, cols, eps);
        return PARALLAX_STATUS_OK;
    }
#endif
    rmsnorm_scalar_fallback(x, w, out, rows, cols, eps);
    return PARALLAX_STATUS_OK;
}
