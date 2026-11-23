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
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * (size_t)cols;
        float *orow = out + (size_t)r * (size_t)cols;
        __m512 sum_vec = _mm512_setzero_ps();
        int c = 0;
        for (; cols - c >= 16; c += 16) {
            __m512 v = _mm512_loadu_ps(xr + c);
            sum_vec = _mm512_fmadd_ps(v, v, sum_vec);
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, sum_vec);
        double ssq = 0.0;
        for (int k = 0; k < 16; ++k) ssq += (double)tmp[k];
        for (; c < cols; ++c) ssq += (double)(xr[c] * xr[c]);
        float rms = 1.0f / sqrtf((float)(ssq / cols) + eps);
        __m512 rms_vec = _mm512_set1_ps(rms);
        c = 0;
        for (; cols - c >= 16; c += 16) {
            __m512 xv = _mm512_loadu_ps(xr + c);
            __m512 wv = _mm512_loadu_ps(w + c);
            _mm512_storeu_ps(orow + c, _mm512_mul_ps(_mm512_mul_ps(xv, rms_vec), wv));
        }
        for (; c < cols; ++c) orow[c] = xr[c] * rms * w[c];
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
        float rms = 1.0f / sqrtf((float)(ssq / (double)cols_s) + eps);
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
