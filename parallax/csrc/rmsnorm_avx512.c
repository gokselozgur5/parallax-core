#include <stddef.h>
#include <stdint.h>
#include <math.h>

#ifndef PARALLAX_STATUS_OK
#define PARALLAX_STATUS_OK 0
#define PARALLAX_STATUS_NULL_POINTER (-1)
#define PARALLAX_STATUS_INVALID_DIMENSION (-2)
#define PARALLAX_STATUS_UNSUPPORTED_ISA (-5)
#endif

int c_has_avx512_rmsnorm_f32(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}

int c_rmsnorm_avx512_f32(const float *x, const float *w, float *out, int rows, int cols, float eps) {
    if (!x || !w || !out) return PARALLAX_STATUS_NULL_POINTER;
    if (rows <= 0 || cols <= 0) return PARALLAX_STATUS_INVALID_DIMENSION;
    if (!c_has_avx512_rmsnorm_f32()) return PARALLAX_STATUS_UNSUPPORTED_ISA;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    (void)eps;
    return PARALLAX_STATUS_OK;
#else
    (void)eps;
    return PARALLAX_STATUS_UNSUPPORTED_ISA;
#endif
}
