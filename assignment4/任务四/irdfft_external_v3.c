/* ========================================================================
 * irdfft_external_v3.c — 逆向 irdFFT 调度引擎 (v3, DIF 结构不变)
 *
 * 与 irdfft_external.c 的区别: M=8/4/2/1 四级走 codelet 特化
 * (irdfft_butterfly_v3.c), M>=16 走通用蝶形 irdfft_butterfly_compute;
 * 归一化循环 LMUL 提升到 m8 (v3 微调)。
 * ======================================================================== */
#include <stdint.h>
#include <riscv_vector.h>

extern void irdfft_butterfly_compute(float *Y, const float *W_re, const float *W_im,
                                     int N, int half_step, int base);
extern void irdfft_stage_M1(float *Y, int N);
extern void irdfft_stage_M2(float *Y, int N);
extern void irdfft_stage_M4(float *Y, int N);
extern void irdfft_stage_M8(float *Y, int N);

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

void execute_irdfft_v3_f32(float *Y, const float *W_re, const float *W_im, int N) {
    for (int M = N / 2; M >= 1; M >>= 1) {
        if (M == 8) {
            irdfft_stage_M8(Y, N);
        } else if (M == 4) {
            irdfft_stage_M4(Y, N);
        } else if (M == 2) {
            irdfft_stage_M2(Y, N);
        } else if (M == 1) {
            irdfft_stage_M1(Y, N);
        } else {
            for (int base = 0; base < N; base += 2 * M)
                irdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
        }
    }

    bit_reverse_permutation(Y, N);

    float inv_N = 1.0f / (float)N;
    int i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m8(N - i);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(&Y[i], vl);
        v = __riscv_vfmul_vf_f32m8(v, inv_N, vl);
        __riscv_vse32_v_f32m8(&Y[i], v, vl);
        i += (int)vl;
    }
}
