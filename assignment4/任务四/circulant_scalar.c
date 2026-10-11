/* ========================================================================
 * circulant_scalar.c — 循环卷积算子 (标量 FFT 实现, 主机 x86 校验用)
 *
 * 提供 circulant_op.h 的标量实现:
 *   circulant_op_init(N) : 预计算 N 点旋转因子表
 *   circulant_op_f32(w,x,y,N) : y = irdFFT(rdFFT(w) .* rdFFT(x))
 *
 * 本文件从任务一纯 C 实现 (pure_clanguage/irdfft-main.c) 提取并封装,
 * 该实现已验证: FFT 往返误差 ~1e-7, 与 torch.fft 结果一致。
 * 第三步会用任务一的 RVV 实现替换本文件, 接口不变。
 * ======================================================================== */
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "circulant_op.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 缓存的旋转因子表 */
static int    g_N = 0;
static float *g_W_re = NULL, *g_W_im = NULL;    /* 正向 rdFFT 用 */
static float *g_IW_re = NULL, *g_IW_im = NULL;  /* 逆向 irdFFT 用 */

/* ---- 位反转置换 ---- */
static void bit_reverse_permute_float(float *arr, int N) {
  int bits = 0;
  int tmp = N;
  while (tmp > 1) { bits++; tmp >>= 1; }
  for (int i = 0; i < N; i++) {
    int j = 0;
    for (int b = 0; b < bits; b++) j = (j << 1) | ((i >> b) & 1);
    if (j > i) { float t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
  }
}

static void generate_rdfft_twiddle_factors(float *W_re, float *W_im, int N) {
  int max_k = N / 2;
  for (int k = 0; k < max_k; k++) {
    double angle = -2.0 * M_PI * (double)k / (double)N;
    W_re[k] = (float)cos(angle);
    W_im[k] = (float)sin(angle);
  }
}

static void generate_irdfft_twiddle_factors(float *W_re, float *W_im, int N) {
  int max_k = N / 2;
  for (int k = 0; k < max_k; k++) {
    double angle = 2.0 * M_PI * (double)k / (double)N;
    W_re[k] = (float)cos(angle);
    W_im[k] = (float)sin(angle);
  }
}

/* ---- 正向 rdFFT ---- */
static void rdfft_butterfly_compute(float *Y, const float *W_re, const float *W_im,
                                    int N, int M, int base) {
  int twiddle_stride = N / (2 * M);
  {
    float E_re = Y[base];
    float O_re = Y[base + M];
    Y[base] = E_re + O_re;
    Y[base + M] = E_re - O_re;
  }
  if (M >= 2) {
    int k = M / 2;
    float E_re = Y[base + k];
    float O_re = Y[base + M + k];
    Y[base + k] = E_re;
    Y[base + 2 * M - k] = -O_re;
  }
  for (int k = 1; k < M / 2; k++) {
    float e_re = Y[base + k];
    float e_im = Y[base + M - k];
    float o_re = Y[base + M + k];
    float o_im = Y[base + 2 * M - k];
    int tw_idx = k * twiddle_stride;
    float w_re = W_re[tw_idx];
    float w_im = W_im[tw_idx];
    float t_re = w_re * o_re - w_im * o_im;
    float t_im = w_re * o_im + w_im * o_re;
    float out_xk_re = e_re + t_re;
    float out_xk_im = e_im + t_im;
    float out_xmk_re = e_re - t_re;
    float out_xmk_im = t_im - e_im;
    Y[base + k] = out_xk_re;
    Y[base + M + k] = out_xmk_im;
    Y[base + M - k] = out_xmk_re;
    Y[base + 2 * M - k] = out_xk_im;
  }
}

static void execute_rdfft_f32(float *Y, const float *W_re, const float *W_im, int N) {
  bit_reverse_permute_float(Y, N);
  for (int M = 1; M < N; M <<= 1)
    for (int base = 0; base < N; base += (2 * M))
      rdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
}

/* ---- 逆向 irdFFT ---- */
static void irdfft_butterfly_compute(float *Y, const float *W_re, const float *W_im,
                                     int N, int M, int base) {
  int twiddle_stride = N / (2 * M);
  {
    float X0 = Y[base];
    float XM = Y[base + M];
    Y[base] = X0 + XM;
    Y[base + M] = X0 - XM;
  }
  if (M >= 2) {
    int k = M / 2;
    float X_re = Y[base + k];
    float X_im = Y[base + 2 * M - k];
    Y[base + k] = 2.0f * X_re;
    Y[base + 2 * M - k] = -2.0f * X_im;
  }
  for (int k = 1; k < M / 2; k++) {
    float a = Y[base + k];
    float b = Y[base + 2 * M - k];
    float c = Y[base + M - k];
    float d = Y[base + M + k];
    int tw_idx = k * twiddle_stride;
    float w_re = W_re[tw_idx];
    float w_im = W_im[tw_idx];
    float diff_re = a - c;
    float diff_im = b + d;
    Y[base + k] = a + c;
    Y[base + M - k] = b - d;
    Y[base + M + k] = diff_re * w_re - diff_im * w_im;
    Y[base + 2 * M - k] = diff_re * w_im + diff_im * w_re;
  }
}

static void execute_irdfft_f32(float *Y, const float *W_re, const float *W_im, int N) {
  for (int M = N / 2; M >= 1; M >>= 1)
    for (int base = 0; base < N; base += (2 * M))
      irdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
  bit_reverse_permute_float(Y, N);
  float inv_N = 1.0f / (float)N;
  for (int i = 0; i < N; i++) Y[i] *= inv_N;
}

static void half_complex_mul(const float *X, const float *Y, float *Z, int N) {
  Z[0] = X[0] * Y[0];
  Z[N / 2] = X[N / 2] * Y[N / 2];
  for (int k = 1; k < N / 2; k++) {
    float a = X[k], b = X[N - k], c = Y[k], d = Y[N - k];
    Z[k] = a * c - b * d;
    Z[N - k] = a * d + b * c;
  }
}

/* ---- 对外接口 ---- */
void circulant_op_init(int N) {
  if (g_N == N && g_W_re) return;
  free(g_W_re); free(g_W_im); free(g_IW_re); free(g_IW_im);
  g_N = N;
  g_W_re  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_W_im  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_IW_re = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_IW_im = (float *)malloc((size_t)(N / 2) * sizeof(float));
  generate_rdfft_twiddle_factors(g_W_re, g_W_im, N);
  generate_irdfft_twiddle_factors(g_IW_re, g_IW_im, N);
}

/* ---- v3 优化: 权重频谱缓存 (与 RVV 版 irdfft_circulant.c 同设计) ----
 * 权重在推理中固定, 按指针缓存其 rdFFT 频谱, 首次计算后直接复用;
 * 缓存频谱只读, 频域乘结果写入独立的 buf_y (不可覆盖缓存)。 ---- */
typedef struct { const float *w; float *W_hc; } WCached;

static WCached *g_wcache = NULL;
static int g_wcache_n = 0, g_wcache_cap = 0;

static const float *wspectrum_get(const float *w, const float *W_re,
                                  const float *W_im, int N) {
  for (int i = 0; i < g_wcache_n; i++)
    if (g_wcache[i].w == w) return g_wcache[i].W_hc;
  if (g_wcache_n == g_wcache_cap) {
    int nc = g_wcache_cap ? g_wcache_cap * 2 : 128;
    g_wcache = (WCached *)realloc(g_wcache, (size_t)nc * sizeof(WCached));
    g_wcache_cap = nc;
  }
  float *W_hc = (float *)malloc((size_t)N * sizeof(float));
  memcpy(W_hc, w, (size_t)N * sizeof(float));
  execute_rdfft_f32(W_hc, W_re, W_im, N);
  g_wcache[g_wcache_n].w = w;
  g_wcache[g_wcache_n].W_hc = W_hc;
  g_wcache_n++;
  return W_hc;
}

void circulant_op_f32(const float *w, const float *x, float *y, int N) {
  if (g_N != N) circulant_op_init(N);
  const float *buf_w = wspectrum_get(w, g_W_re, g_W_im, N);  /* 缓存, 只读 */
  float *buf_x = (float *)malloc((size_t)N * sizeof(float));
  float *buf_y = (float *)malloc((size_t)N * sizeof(float));
  memcpy(buf_x, x, (size_t)N * sizeof(float));
  execute_rdfft_f32(buf_x, g_W_re, g_W_im, N);
  half_complex_mul(buf_w, buf_x, buf_y, N);
  execute_irdfft_f32(buf_y, g_IW_re, g_IW_im, N);
  memcpy(y, buf_y, (size_t)N * sizeof(float));
  free(buf_x);
  free(buf_y);
}
