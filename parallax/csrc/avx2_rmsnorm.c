#include <stddef.h>
#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma")
#endif
#include <immintrin.h>

int c_has_avx2_rmsnorm(void) {
    return 1;
}

void c_rmsnorm_avx2_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    if (!x || !w || !o || r <= 0 || c <= 0) return;
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
#else
int c_has_avx2_rmsnorm(void) {
    return 0;
}

void c_rmsnorm_avx2_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    (void)x; (void)w; (void)o; (void)r; (void)c; (void)eps;
}
#endif
