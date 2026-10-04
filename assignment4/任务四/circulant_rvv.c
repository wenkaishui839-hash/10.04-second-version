/* ========================================================================
 * circulant_rvv.c — 循环卷积算子 (RVV 实现, 第三步 Spike 跑分用)
 *
 * 提供与 circulant_op.h 完全相同的接口:
 *   circulant_op_init(N)  : 预计算 N 点旋转因子表
 *   circulant_op_f32(...) : y = C(w) * x (循环卷积)
 *
 * 内部调用任务一的 RVV 实现 circulant_matmul_rvv_f32, 它与标量版
 * circulant_scalar.c 唯一的区别是 FFT 由 RISC-V 向量指令完成。
 * 推理主逻辑 (inference.c) 无需任何改动, 编译时用本文件替换
 * circulant_scalar.c 并额外链接任务一的 7 个算子源文件即可。
 * ======================================================================== */
#include <stdlib.h>

#include "circulant_op.h"

/* 任务一 RVV 算子 (位于上级目录 dachuang/, 与 rdfft_*.c / irdfft_*.c 一起链接) */
extern void generate_rdfft_twiddle_factors(float *W_re, float *W_im, int N);
extern void circulant_matmul_rvv_f32(const float *w, const float *x, float *y,
                                     const float *W_re, const float *W_im, int N);

/* 缓存的旋转因子表 (RVV 版仅一张正向表, irdFFT 蝶形内部取共轭复用) */
static int    g_N = 0;
static float *g_W_re = NULL, *g_W_im = NULL;

void circulant_op_init(int N) {
  if (g_N == N && g_W_re) return;
  free(g_W_re);
  free(g_W_im);
  g_N = N;
  g_W_re = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_W_im = (float *)malloc((size_t)(N / 2) * sizeof(float));
  generate_rdfft_twiddle_factors(g_W_re, g_W_im, N);
}

void circulant_op_f32(const float *w, const float *x, float *y, int N) {
  if (g_N != N) circulant_op_init(N);
  circulant_matmul_rvv_f32(w, x, y, g_W_re, g_W_im, N);
}
