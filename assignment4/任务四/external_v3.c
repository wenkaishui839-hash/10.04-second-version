/* ========================================================================
 * external_v3.c — 正向 rdFFT 调度引擎 (v3, 原论文 DIT 共轭对结构不变)
 *
 * 与 external.c 的唯一区别: M=1/2/4/8 四级改走 codelet 特化
 * (rdfft_butterfly_v3.c), M>=16 仍走通用蝶形 rdfft_butterfly_compute。
 * 位反转置换与 v2 相同 (置换表预计算 + 标量 swap, 零额外缓冲)。
 * ======================================================================== */
#include <stdint.h>

extern void rdfft_butterfly_compute(float *Y, const float *W_re, const float *W_im,
                                    int N, int half_step, int base);
extern void rdfft_stage_M1(float *Y, int N);
extern void rdfft_stage_M2(float *Y, int N);
extern void rdfft_stage_M4(float *Y, int N);
extern void rdfft_stage_M8(float *Y, int N);

/* 位反转置换表 (v2 优化: 预计算一次 + 查表交换, 标量原地 swap) */
static uint32_t g_rev_table[1024];
static int      g_rev_table_n = 0;

static void bit_reverse_permutation(float *Y, int N) {
    if (g_rev_table_n != N) {
        int bits = 0;
        for (int t = N >> 1; t > 0; t >>= 1) bits++;
        for (int i = 0; i < N; i++) {
            uint32_t j = 0;
            for (int b = 0; b < bits; b++) j = (j << 1) | ((uint32_t)(i >> b) & 1u);
            g_rev_table[i] = j;
        }
        g_rev_table_n = N;
    }
    for (int i = 0; i < N; i++) {
        uint32_t j = g_rev_table[i];
        if (i < (int)j) {
            float temp = Y[i];
            Y[i] = Y[j];
            Y[j] = temp;
        }
    }
}

void execute_rdfft_v3_f32(float *Y, const float *W_re, const float *W_im, int N) {
    bit_reverse_permutation(Y, N);

    for (int M = 1; M < N; M <<= 1) {
        if (M == 1) {
            rdfft_stage_M1(Y, N);
        } else if (M == 2) {
            rdfft_stage_M2(Y, N);
        } else if (M == 4) {
            rdfft_stage_M4(Y, N);
        } else if (M == 8) {
            rdfft_stage_M8(Y, N);
        } else {
            for (int base = 0; base < N; base += 2 * M)
                rdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
        }
    }
}
