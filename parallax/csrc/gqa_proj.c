static void proj_f32(const float *x, const float *w, float *out, int rows, int dim, int cols) {
    for (int r = 0; r < rows; ++r) {
        float *o = out + (long)r * cols;
        for (int c = 0; c < cols; ++c) o[c] = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float xv = x[(long)r * dim + d];
            const float *wr = w + (long)d * cols;
            for (int c = 0; c < cols; ++c) o[c] += xv * wr[c];
        }
    }
}

/* Q: [rows, n_heads*hd]; K,V: [rows, n_kv*hd] replicated to [rows, n_heads*hd]. */
int c_gqa_proj_f32(const float *x, const float *wq, const float *wk, const float *wv,
                   float *q, float *k, float *v, float *tmp,
                   int rows, int dim, int n_heads, int n_kv, int hd) {
    if (!x || !wq || !wk || !wv || !q || !k || !v || !tmp) return -1;
    if (rows <= 0 || dim <= 0 || hd <= 0 || n_kv <= 0 || n_heads < n_kv || n_heads % n_kv) return -1;
    int rep = n_heads / n_kv, qc = n_heads * hd, kc = n_kv * hd;
    proj_f32(x, wq, q, rows, dim, qc);
    float *outs[2] = {k, v};
    const float *ws[2] = {wk, wv};
    for (int t = 0; t < 2; ++t) {
        proj_f32(x, ws[t], tmp, rows, dim, kc);
        for (int r = 0; r < rows; ++r)
            for (int h = 0; h < n_heads; ++h)
                for (int i = 0; i < hd; ++i)
                    outs[t][(long)r * qc + h * hd + i] = tmp[(long)r * kc + (h / rep) * hd + i];
    }
    return 0;
}
