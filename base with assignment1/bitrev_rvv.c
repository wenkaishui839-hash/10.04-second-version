/* ========================================================================
 * bitrev_rvv.c — 位反转置换的三种实现 (rdFFT v2 结构内优化)
 *
 * 原论文共轭对结构不变 (DIT: 位反转 → 逐级共轭对蝶形), 只优化位反转本身:
 *   mode 0: j 增量算法 (与 external.c 内 static 版一致, 基线)
 *   mode 1: 预计算置换表 + 查表交换 (无每元素 j 增量计算)
 *   mode 2: RVV 双缓冲 (vluxei32 收集 + 连续写回, 需 1 个 N 元素临时缓冲)
 *
 * 注意: 早前在 GCC 15.1.0 / Spike 上 vluxei32 曾产生 NaN 已回退; mode 2
 * 在 clang / 真板上重新实验, fft_v2_bench.c 内带频点正确性检查兜底。
 * ======================================================================== */
#include <stdint.h>
#include <string.h>
#include <riscv_vector.h>

/* 预计算位反转索引表 rev[i] = bit_reverse(i), 一次性调用 */
void bitrev_table_build(uint32_t* rev, int N) {
    int bits = 0;
    for (int t = N >> 1; t > 0; t >>= 1) bits++;
    for (int i = 0; i < N; i++) {
        uint32_t j = 0;
        for (int b = 0; b < bits; b++) j = (j << 1) | ((uint32_t)(i >> b) & 1u);
        rev[i] = j;
    }
}

/* mode 0: j 增量算法 (基线, 与 external.c 的 bit_reverse_permutation 同逻辑) */
void bitrev_swap_jincrement(float* Y, int N) {
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            float temp = Y[i];
            Y[i] = Y[j];
            Y[j] = temp;
        }
        int k = N / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

/* mode 1: 预计算置换表 + 查表交换 (仅 i < rev[i] 时交换一次) */
void bitrev_swap_table(float* Y, const uint32_t* rev, int N) {
    for (int i = 0; i < N; i++) {
        uint32_t j = rev[i];
        if (i < (int)j) {
            float temp = Y[i];
            Y[i] = Y[j];
            Y[j] = temp;
        }
    }
}

/* mode 2: RVV 双缓冲置换 out[i] = in[rev[i]] (原地安全, 全向量化;
 * scratch 由调用者提供, 长度 >= N floats)
 *
 * ⚠ 实测不可用 (2026-09): clang + CanMV K230 (C908) 上 vluxei32 触发非对齐
 * 访存异常 (cause=4 load address misaligned, SIGBUS); 早前 GCC 15.1 + Spike
 * 上则产生 NaN。两个独立组合均失败 → 离散索引向量访存在该平台上不可靠,
 * 位反转向量化改走 mode 1 (置换表查表标量交换)。本函数仅作记录保留。 */
void bitrev_permute_rvv(const float* Y_in, float* Y_out, const uint32_t* rev, int N) {
    int i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vuint32m1_t vidx = __riscv_vle32_v_u32m1(&rev[i], vl);
        vfloat32m1_t v = __riscv_vluxei32_v_f32m1(Y_in, vidx, vl);
        __riscv_vse32_v_f32m1(&Y_out[i], v, vl);
        i += (int)vl;
    }
}
