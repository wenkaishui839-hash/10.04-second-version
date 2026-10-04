#include <stdint.h>

extern void rdfft_butterfly_compute(float* Y, const float* W_re, const float* W_im,
                                     int N, int half_step, int base);

/**
 * 位反转置换 (对 N 个实数做索引层面的重排)
 *
 * 这是 DIT FFT 的标准前置步骤。在 rdFFT 中，位反转直接作用于 N 个实数索引，
 * 因为"隐式复数编码"将实部/虚部分别映射到 k 和 N-k 位置，
 * 位反转会自然保持这种镜像关系在组内的正确性。
 */
static void bit_reverse_permutation(float* Y, int N) {
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

/**
 * 真·rdFFT：完全实域原地快速傅里叶变换
 *
 * 算法路线（共轭对蝶形 + 隐式复数编码）:
 *   1. 位反转：将 N 个实数按 bit-reverse 重排 (O(N))
 *   2. 逐级共轭对蝶形：从 half_step=1 到 N/2，每级组内以
 *      隐式编码 (Re@k, Im@M-k) 完成 4 读 4 写的纯实数蝶形 (O(N log N))
 *   3. 直接产出：计算结束时 Y 即为半复数格式的频域结果，
 *      无需任何解包步骤、无额外缓存分配
 *
 * 半复数输出格式:
 *   Y[0]   = Re(F[0])    直流分量
 *   Y[N/2] = Re(F[N/2])  奈奎斯特分量
 *   Y[k]   = Re(F[k]),  Y[N-k] = Im(F[k])   (0 < k < N/2)
 *
 * @param Y    实数数组，长度 N。输入为 N 个实数时域采样，输出为频域隐式编码。
 * @param W_re 旋转因子实部表 W[k] = cos(-2*pi*k/N), 长度 N/2
 * @param W_im 旋转因子虚部表 W[k] = sin(-2*pi*k/N), 长度 N/2
 * @param N    信号长度 (必须是 2 的幂次方)
 */
void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N) {

    // ==========================================================
    // 第一步：位反转 - O(N)
    // 将 N 个实数时域采样按位反转索引重排
    // ==========================================================
    bit_reverse_permutation(Y, N);

    // ==========================================================
    // 第二步：逐级共轭对蝶形 - O(N log N)
    //
    // half_step 从 1 翻倍到 N/2，每组大小 step = 2 * half_step
    // 组内以隐式编码 (模 M = half_step) 进行共轭对蝶形:
    //   E 子块: Re@base+k, Im@base+M-k
    //   O 子块: Re@base+M+k, Im@base+2M-k
    //
    // 全程 4 读 4 写，零缓存，不经过任何 N+2 中间态
    // ==========================================================
    for (int half_step = 1; half_step < N; half_step *= 2) {
        int step = 2 * half_step;

        for (int base = 0; base < N; base += step) {
            rdfft_butterfly_compute(Y, W_re, W_im, N, half_step, base);
        }
    }

    // 结束！Y 已经是半复数格式的频域结果。
    // 全程无解包、无临时缓存、无复数张量分配。
    // 输入 N 实 → 输出 N 实，真·零缓存原地。
}
