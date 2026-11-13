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
#else
int c_has_avx2_rmsnorm(void) {
    return 0;
}

void c_rmsnorm_avx2_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    (void)x; (void)w; (void)o; (void)r; (void)c; (void)eps;
}
#endif
