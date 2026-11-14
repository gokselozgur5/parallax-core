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
