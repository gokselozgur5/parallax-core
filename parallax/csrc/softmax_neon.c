#include "parallax_simd.h"
#include <math.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_softmax_neon_f32(const float *in, float *out, int rows, int cols) {
    (void)in; (void)out; (void)rows; (void)cols;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else

int c_softmax_neon_f32(const float *in, float *out, int rows, int cols) {
    if (!in || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (rows <= 0 || cols <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    return PARALLAX_STATUS_OK;
}
#endif
