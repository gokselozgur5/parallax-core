#include <math.h>

void c_rope_f32(float *x, int T, int H, int D, int start_pos, float theta, int interleaved) {
    if (!x || T <= 0 || H <= 0 || D <= 0 || (D % 2 != 0)) return;
    int half = D / 2;
    for (int t = 0; t < T; ++t) {
        float pos = (float)(start_pos + t);
        for (int i = 0; i < half; ++i) {
            float angle = pos / powf(theta, (float)(2 * i) / (float)D);
            float c = cosf(angle), s = sinf(angle);
            for (int h = 0; h < H; ++h) {
                float *head = x + (t * H + h) * D;
                int idx1 = interleaved ? 2 * i : i;
                int idx2 = interleaved ? 2 * i + 1 : half + i;
                float v1 = head[idx1], v2 = head[idx2];
                head[idx1] = v1 * c - v2 * s;
                head[idx2] = v1 * s + v2 * c;
            }
        }
    }
}
