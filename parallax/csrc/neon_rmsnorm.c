#include <stddef.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

int c_has_neon_rmsnorm(void) {
    return 1;
}
#else
int c_has_neon_rmsnorm(void) {
    return 0;
}

void c_rmsnorm_neon_f32(const float *x, const float *w, float *o, int r, int c, float eps) {
    (void)x; (void)w; (void)o; (void)r; (void)c; (void)eps;
}
#endif
