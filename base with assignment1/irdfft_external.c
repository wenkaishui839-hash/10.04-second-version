/* ========================================================================
 * irdfft_external.c — 逆向 irdFFT 顶层调度引擎 (RVV, DIF 结构)
 * 
 * 调度流程 (与正变换 DIT 互为镜像):
 *   1. 逐级蝶形: half_step 从 N/2 递减到 1, 调用 irdfft_butterfly_compute
 *      (DIF 结构, 每一级消耗上一级产生的半复数配对布局)
 *   2. 位反转: 得到自然顺序的时域序列 (O(N))
 *   3. 归一化: 每级蝶形放大 2 倍, log2(N) 级共放大 N 倍,
 *      最后全数组统一向量化乘以 1/N (vfmul_vf)
 *
 * 注意: 不能照搬任务书"先位反转、再 M 从小到大"的 DIT 写法 —
 *   位反转会打乱半复数频谱的共轭对布局, 导致蝶形读到非法配对
 *   (已在 N=4 上手工推演证实整体错误)。DIF 才是该编码的正确逆运算。
 *
 * 输入: Y 为半复数格式频谱 (Y[k]=Re, Y[N-k]=Im, DC/Nyquist 纯实数)
 * 输出: Y 为时域实数序列 x[0..N-1], 已归一化
 * ======================================================================== */
#include <stdint.h>
#include <riscv_vector.h>

extern void irdfft_butterfly_compute(float* Y, const float* W_re, const float* W_im,
                                     int N, int half_step, int base);

/* 位反转置换 (对 N 个实数做索引层面的重排, O(N))
 * 与 external.c 中的实现逻辑一致; 设为 static 避免与正变换链接冲突。
 * v2 优化: 置换表预计算一次 + 查表交换 (见 external.c 注释)。 */
static uint32_t g_rev_table[1024];
static int      g_rev_table_n = 0;

static void bit_reverse_permutation(float* Y, int N) {
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

/**
 * 执行完整逆向 irdFFT (原地)
 *
 * @param Y    半复数格式频谱数组, 长度 N; 输出为时域实数序列
 * @param W_re 旋转因子实部表 (与正变换共用同一张表, 长度为 N/2)
 * @param W_im 旋转因子虚部表
 * @param N    信号长度 (必须是 2 的幂次方)
 */
void execute_irdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N) {

    /* 第一步: 逐级蝶形, half_step 从 N/2 递减到 1 - O(N log N) */
    for (int half_step = N / 2; half_step >= 1; half_step /= 2) {
        int step = 2 * half_step;

        for (int base = 0; base < N; base += step) {
            irdfft_butterfly_compute(Y, W_re, W_im, N, half_step, base);
        }
    }

    /* 第二步: 位反转 - O(N) */
    bit_reverse_permutation(Y, N);

    /* 第三步: 向量化归一化 (乘以 1/N) - O(N) */
    float inv_N = 1.0f / (float)N;
    int i = 0;
    while (i < N) {
        size_t vl = __riscv_vsetvl_e32m1(N - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(&Y[i], vl);
        v = __riscv_vfmul_vf_f32m1(v, inv_N, vl);
        __riscv_vse32_v_f32m1(&Y[i], v, vl);
        i += (int)vl;
    }
}
