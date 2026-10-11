/* ========================================================================
 * external_staged.c — rdFFT v2 调度引擎 (原论文共轭对结构不变)
 *
 * 与 external.c 相同: 位反转 → 逐级共轭对蝶形 (half_step 1 → N/2),
 * 区别: 蝶形用逐级连续旋转因子表 (rdfft_butterfly_staged), 位反转可选
 * 三种实现 (bitrev_mode 0/1/2, 见 bitrev_rvv.c)。
 *
 * @param bitrev_mode  0 = j 增量 (基线); 1 = 置换表查表交换;
 *                     2 = RVV 双缓冲 (vluxei32, 需 scratch 缓冲)
 * @param scratch      仅 mode 2 使用, 长度 >= N floats (可为 NULL 其余模式)
 * ======================================================================== */
#include <stdint.h>
#include <string.h>

extern void rdfft_butterfly_staged(float* Y, const float* W_re, const float* W_im,
                                   int N, int half_step, int base);
extern void bitrev_swap_jincrement(float* Y, int N);
extern void bitrev_swap_table(float* Y, const uint32_t* rev, int N);
extern void bitrev_permute_rvv(const float* Y_in, float* Y_out,
                               const uint32_t* rev, int N);

void execute_rdfft_staged_f32(float* Y, const float* W_re, const float* W_im,
                              int N, int bitrev_mode,
                              float* scratch, const uint32_t* rev) {
    /* 第一步: 位反转 (三种实现, 数学等价) */
    if (bitrev_mode == 2) {
        bitrev_permute_rvv(Y, scratch, rev, N);
        memcpy(Y, scratch, (size_t)N * sizeof(float));
    } else if (bitrev_mode == 1) {
        bitrev_swap_table(Y, rev, N);
    } else {
        bitrev_swap_jincrement(Y, N);
    }

    /* 第二步: 逐级共轭对蝶形 (与 external.c 相同的 DIT 调度) */
    for (int half_step = 1; half_step < N; half_step *= 2) {
        int step = 2 * half_step;

        for (int base = 0; base < N; base += step) {
            rdfft_butterfly_staged(Y, W_re, W_im, N, half_step, base);
        }
    }
}
