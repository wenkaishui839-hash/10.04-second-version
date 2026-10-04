void rdfft_butterfly_compute_scalar(float* Y, const float* W_re, const float* W_im,
                              int N, int half_step, int base) {
    int M = half_step;
    int twiddle_stride = N / (2 * M);

    // ==========================================================
    // Case 1: k = 0 — DC 蝶形
    // ==========================================================
    float E_re_0 = Y[base];
    float O_re_0 = Y[base + M];
    Y[base]     = E_re_0 + O_re_0; 
    Y[base + M] = E_re_0 - O_re_0;

    // ==========================================================
    // Case 2: k = M/2 — Nyquist 蝶形
    // ==========================================================
    if (M >= 2) {
        int k = M / 2;
        float E_re_M2 = Y[base + k];
        float O_re_M2 = Y[base + M + k];
        Y[base + k]         = E_re_M2;
        Y[base + 2 * M - k] = -O_re_M2;
    }

    // ==========================================================
    // Case 3: k = 1 .. M/2-1 — 常规共轭对蝶形 (纯标量循环)
    // ==========================================================
    for (int k = 1; k < M / 2; k++) {
        // 抓取 4 个点的物理数据
        float e_re = Y[base + k];
        float e_im = Y[base + M - k];
        float o_re = Y[base + M + k];
        float o_im = Y[base + 2 * M - k];

        // 获取旋转因子
        int w_idx = k * twiddle_stride;
        float w_re = W_re[w_idx];
        float w_im = W_im[w_idx];

        // 纯数学复数乘法
        float t_re = w_re * o_re - w_im * o_im;
        float t_im = w_re * o_im + w_im * o_re;

        // 共轭对蝶形加减
        float out_xk_re  = e_re + t_re;
        float out_xk_im  = e_im + t_im;
        float out_xmk_re = e_re - t_re;
        float out_xmk_im = t_im - e_im; // 注意这里的顺序

        // 极其精准的原位写回
        Y[base + k]         = out_xk_re;
        Y[base + M + k]     = out_xmk_im;
        Y[base + M - k]     = out_xmk_re;
        Y[base + 2 * M - k] = out_xk_im;
    }
}