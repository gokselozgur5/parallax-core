#include <stddef.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

int c_has_neon_rmsnorm(void) {
    return 1;
}

void c_rmsnorm_neon_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    if (!x || !w || !o || r <= 0 || c <= 0) return;
    const size_t rows = (size_t)r, cols = (size_t)c;
    for (size_t row = 0; row < rows; ++row) {
        const float *xr = x + row * cols;
        float *orow = o + row * cols;
        float64x2_t sv0 = vdupq_n_f64(0.0), sv1 = vdupq_n_f64(0.0);
        size_t col = 0;
        for (; cols - col >= 4; col += 4) {
            float32x4_t xv = vld1q_f32(xr + col);
            float64x2_t x0 = vcvt_f64_f32(vget_low_f32(xv));
            float64x2_t x1 = vcvt_f64_f32(vget_high_f32(xv));
            sv0 = vmlaq_f64(sv0, x0, x0);
            sv1 = vmlaq_f64(sv1, x1, x1);
        }
        double t0[2], t1[2];
        vst1q_f64(t0, sv0); vst1q_f64(t1, sv1);
        double sum_sq = (t0[0] + t0[1]) + (t1[0] + t1[1]);
        for (; col < cols; ++col) sum_sq += (double)xr[col] * (double)xr[col];
        float inv_rms = 1.0f / (float)sqrt(sum_sq / (double)cols + (double)eps);
        float32x4_t inv_vec = vdupq_n_f32(inv_rms);
        for (col = 0; cols - col >= 4; col += 4) {
            float32x4_t out_v = vmulq_f32(vmulq_f32(vld1q_f32(xr + col), inv_vec), vld1q_f32(w + col));
            vst1q_f32(orow + col, out_v);
        }
        for (; col < cols; ++col) orow[col] = (xr[col] * inv_rms) * w[col];
    }
}
#else
int c_has_neon_rmsnorm(void) {
    return 0;
}

void c_rmsnorm_neon_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    (void)x; (void)w; (void)o; (void)r; (void)c; (void)eps;
}
#endif
