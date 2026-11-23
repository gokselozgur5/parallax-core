#include <math.h>
#include <stddef.h>
#include <stdint.h>

void c_rope_f32(float *x, int T, int H, int D, int start_pos, float theta, int interleaved) {
    if (!x || T <= 0 || H <= 0 || D <= 0 || (D % 2 != 0)) return;
    const size_t Ts = (size_t)T, Hs = (size_t)H, Ds = (size_t)D;
    const size_t half = Ds / 2;
    for (size_t t = 0; t < Ts; ++t) {
        float pos = (float)((int64_t)start_pos + (int64_t)t);
        for (size_t i = 0; i < half; ++i) {
            float angle = pos / powf(theta, (float)(2 * i) / (float)Ds);
            float c = cosf(angle), s = sinf(angle);
            for (size_t h = 0; h < Hs; ++h) {
                float *head = x + (t * Hs + h) * Ds;
                size_t idx1 = interleaved ? 2 * i : i;
                size_t idx2 = interleaved ? 2 * i + 1 : half + i;
                float v1 = head[idx1], v2 = head[idx2];
                head[idx1] = v1 * c - v2 * s;
                head[idx2] = v1 * s + v2 * c;
            }
        }
    }
}
