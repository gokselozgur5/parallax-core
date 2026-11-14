#include "parallax_simd.h"
#include <math.h>

#if !defined(PARALLAX_HAVE_NEON)
int c_flashattn_neon_f32(const float *q, const float *k, const float *v,
                         float *out, int seq_len, int head_dim,
                         float scale, int is_causal) {
    (void)q; (void)k; (void)v; (void)out; (void)seq_len;
    (void)head_dim; (void)scale; (void)is_causal;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
}
#else
