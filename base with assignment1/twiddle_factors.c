#define _USE_MATH_DEFINES
#include <math.h>

/**
 * 生成 N 点 DFT 旋转因子表
 *
 * 公式: W[k] = e^(-j * 2 * pi * k / N),  k = 0 .. N/2-1
 *
 * 在真·rdFFT 共轭对蝶形中，第 half_step 级的第 k 个蝶形使用:
 *   W_index = k * N / (2 * half_step) = k * twiddle_stride
 * 以 vlse32 strided load 从本表中按 stride 步进读取。
 *
 * @param W_re  预分配的实部数组空间 (大小至少为 N/2)
 * @param W_im  预分配的虚部数组空间 (大小至少为 N/2)
 * @param N     FFT 的总点数 (必须是 2 的幂次方)
 */
void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N) {
    int max_k = N / 2;

    for (int k = 0; k < max_k; k++) {
        double angle = -2.0 * M_PI * (double)k / (double)N;

        W_re[k] = (float)cos(angle);
        W_im[k] = (float)sin(angle);
    }
}

/* ========================================================================
 * 逐级连续旋转因子表 (rdFFT v2 优化, 原论文共轭对结构内优化):
 * 第 s 级 (M = 2^s) 的表紧邻排布, 阶段内偏移 = M/2 - 1 (各阶段表长前缀和),
 * 蝶形内用连续 vle32 替代 vlse32 strided load。
 * 数学与原表完全一致: 第 M 级第 k 个蝶形用 W_{2M}^k = e^{-j2πk/(2M)}。
 * ======================================================================== */
void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N) {
    int off = 0;
    for (int M = 1; M <= N / 2; M *= 2) {
        int len = M / 2;   /* 表长: k = 0 .. M/2-1 (M=1,2 的表未被蝶形使用) */
        for (int k = 0; k < len; k++) {
            double angle = -2.0 * M_PI * (double)k / (double)(2 * M);
            W_re[off + k] = (float)cos(angle);
            W_im[off + k] = (float)sin(angle);
        }
        off += len;
    }
}
