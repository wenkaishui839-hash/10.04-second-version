/* ========================================================================
 * irdfft_circulant.c — 基于 FFT 的 Circulant 矩阵乘法算子 (RVV) 
 *
 * 组合公式: Y = irdFFT(rdFFT(W) .* rdFFT(X))
 *
 * C(w) 是第一列为 w[0..N-1] 的循环矩阵:
 *   [ w0    w1    w2   ... w_{N-1} ]
 *   [ w_{N-1} w0   w1   ... w_{N-2} ]
 *   ...
 * 矩阵向量乘等价于 w 与 x 的循环卷积, 用 FFT 加速为 O(N log N)。
 *
 * 正变换与逆变换共用同一张 twiddle 表 (W_re, W_im),
 * 逆变换在蝶形内部取共轭使用, 省去第二张表的存储与预计算开销。
 *
 * 任务四说明: 本文件 + irdfft_butterfly_compute.c + irdfft_external.c
 * + irdfft_half_complex_mul.c 连同正变换三个文件, 即为 NICE 推理程序
 * 需要链接的算子集合 (不含 main, 可直接链接)。
 * ======================================================================== */
#include <stdlib.h>
#include <string.h>

extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void execute_irdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void half_complex_mul_rvv(const float* restrict A, const float* restrict B,
                                 float* restrict C, int N);

void circulant_matmul_rvv_f32(const float* w, const float* x, float* y,
                              const float* W_re, const float* W_im, int N) {

    /* 缓冲区: 保存 rdFFT 后的半复数, 按需分配 */
    float* W_hc = (float*)malloc(sizeof(float) * N);
    float* X_hc = (float*)malloc(sizeof(float) * N);
    float* Y_hc = (float*)malloc(sizeof(float) * N);

    memcpy(W_hc, w, sizeof(float) * N);
    memcpy(X_hc, x, sizeof(float) * N);

    /* 1. rdFFT 正变换得到半复数 */
    execute_rdfft_rvv_f32(W_hc, W_re, W_im, N);
    execute_rdfft_rvv_f32(X_hc, W_re, W_im, N);

    /* 2. 半复数逐元素相乘 */
    half_complex_mul_rvv(W_hc, X_hc, Y_hc, N);

    /* 3. 逆 rdFFT 得到时域结果 y (共用同一张 twiddle 表) */
    execute_irdfft_rvv_f32(Y_hc, W_re, W_im, N);

    memcpy(y, Y_hc, sizeof(float) * N);

    free(W_hc);
    free(X_hc);
    free(Y_hc);
}
