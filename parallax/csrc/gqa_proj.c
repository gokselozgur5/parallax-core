#include <stddef.h>

static void proj_f32(const float *x, const float *w, float *out, size_t rows, size_t dim, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float *o = out + r * cols;
        for (size_t c = 0; c < cols; ++c) o[c] = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            float xv = x[r * dim + d];
            const float *wr = w + d * cols;
            for (size_t c = 0; c < cols; ++c) o[c] += xv * wr[c];
        }
    }
}

/* Q: [rows, n_heads*hd]; K,V: [rows, n_kv*hd] replicated to [rows, n_heads*hd]. */
int c_gqa_proj_f32(const float *x, const float *wq, const float *wk, const float *wv,
                   float *q, float *k, float *v, float *tmp,
                   int rows, int dim, int n_heads, int n_kv, int hd) {
    if (!x || !wq || !wk || !wv || !q || !k || !v || !tmp) return -1;
    if (rows <= 0 || dim <= 0 || hd <= 0 || n_kv <= 0 || n_heads < n_kv || n_heads % n_kv) return -1;
    const size_t rows_s = (size_t)rows, dim_s = (size_t)dim;
    const size_t nh_s = (size_t)n_heads, nkv_s = (size_t)n_kv, hd_s = (size_t)hd;
    const size_t rep = nh_s / nkv_s, qc = nh_s * hd_s, kc = nkv_s * hd_s;
    proj_f32(x, wq, q, rows_s, dim_s, qc);
    float *outs[2] = {k, v};
    const float *ws[2] = {wk, wv};
    for (size_t t = 0; t < 2; ++t) {
        proj_f32(x, ws[t], tmp, rows_s, dim_s, kc);
        for (size_t r = 0; r < rows_s; ++r)
            for (size_t h = 0; h < nh_s; ++h)
                for (size_t i = 0; i < hd_s; ++i)
                    outs[t][r * qc + h * hd_s + i] = tmp[r * kc + (h / rep) * hd_s + i];
    }
    return 0;
}
