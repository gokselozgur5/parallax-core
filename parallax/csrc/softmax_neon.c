#include "parallax_simd.h"
#include <math.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_softmax_neon_f32(const float *in, float *out, int rows, int cols) {
    (void)in; (void)out; (void)rows; (void)cols;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

static inline float32x4_t poly_exp_vec(float32x4_t x) {
    float32x4_t k = vrndaq_f32(vmulq_n_f32(x, 1.44269504f));
    float32x4_t r = vmlsq_n_f32(x, k, 0.69314718f);
    float32x4_t p = vmlaq_n_f32(vdupq_n_f32(1.0f / 120.0f), r, 1.0f / 720.0f);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 24.0f), r, p);
    p = vmlaq_f32(vdupq_n_f32(1.0f / 6.0f), r, p);
    p = vmlaq_f32(vdupq_n_f32(0.5f), r, p);
    p = vmlaq_f32(vdupq_n_f32(1.0f), r, p);
    p = vmlaq_f32(vdupq_n_f32(1.0f), r, p);
    int32x4_t ik = vaddq_s32(vcvtq_s32_f32(k), vdupq_n_s32(127));
    return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(ik, 23)));
}

int c_softmax_neon_f32(const float *in, float *out, int rows, int cols) {
    if (!in || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (rows <= 0 || cols <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    return PARALLAX_STATUS_OK;
}
#endif
