/* ========================================================================
 * circulant_scalar_v3.c — 循环卷积算子 (纯标量, v3 零 malloc 重构)
 *
 * 与 RVV 版 circulant_v3.c 同设计 (init / prepare / f32 三段式):
 *   - 前向过程零 malloc (常驻 g_scratch + 原地半复数乘 + 注册表兜底)
 *   - FFT 蝶形保持原标量实现逐行不变 (标量基线口径不变)
 *
 * 标量版单独维护正/逆两张旋转因子表 (蝶形内部不做共轭),
 * 与旧版 circulant_scalar.c 的数学完全一致。
 * ======================================================================== */
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "circulant_op.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int    g_N = 0;
static float *g_W_re = NULL, *g_W_im = NULL;    /* 正向 rdFFT 用 */
static float *g_IW_re = NULL, *g_IW_im = NULL;  /* 逆向 irdFFT 用 */
static float *g_scratch = NULL;                 /* x 频谱原地工作区 */
static float *g_w_fallback = NULL;              /* 未注册权重兜底频谱 */

typedef struct { const float *w; float *W_hc; } WEntry;
static WEntry *g_wreg = NULL;
static int g_wreg_n = 0, g_wreg_cap = 0;

/* ---- 位反转置换 (与原标量版一致) ---- */
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

/* ---- 正向 rdFFT (与原标量版逐行一致) ---- */
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

/* ---- 逆向 irdFFT (与原标量版逐行一致) ---- */
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

/* ---- 半复数频域乘 (v3 原地版: B = A .* B, 先读后写) ---- */
static void half_complex_mul_inplace(const float *A, float *B, int N) {
  B[0] = A[0] * B[0];
  B[N / 2] = A[N / 2] * B[N / 2];
  for (int k = 1; k < N / 2; k++) {
    float a = A[k], b = A[N - k], c = B[k], d = B[N - k];
    B[k] = a * c - b * d;
    B[N - k] = a * d + b * c;
  }
}

/* ---- v3 三段式接口 ---- */
void circulant_op_init(int N) {
  if (g_N == N && g_W_re) return;
  free(g_W_re); free(g_W_im); free(g_IW_re); free(g_IW_im);
  free(g_scratch); free(g_w_fallback);
  g_N = N;
  g_W_re  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_W_im  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_IW_re = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_IW_im = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_scratch = (float *)malloc((size_t)N * sizeof(float));
  g_w_fallback = (float *)malloc((size_t)N * sizeof(float));
  generate_rdfft_twiddle_factors(g_W_re, g_W_im, N);
  generate_irdfft_twiddle_factors(g_IW_re, g_IW_im, N);
  g_wreg_n = 0;
}

void circulant_op_prepare_weight(const float *w) {
  if (g_N == 0 || !w) return;
  for (int i = 0; i < g_wreg_n; i++)
    if (g_wreg[i].w == w) return;
  if (g_wreg_n == g_wreg_cap) {
    int nc = g_wreg_cap ? g_wreg_cap * 2 : 512;
    g_wreg = (WEntry *)realloc(g_wreg, (size_t)nc * sizeof(WEntry));
    g_wreg_cap = nc;
  }
  float *W_hc = (float *)malloc((size_t)g_N * sizeof(float));
  memcpy(W_hc, w, (size_t)g_N * sizeof(float));
  execute_rdfft_f32(W_hc, g_W_re, g_W_im, g_N);
  g_wreg[g_wreg_n].w = w;
  g_wreg[g_wreg_n].W_hc = W_hc;
  g_wreg_n++;
}

void circulant_op_f32(const float *w, const float *x, float *y, int N) {
  if (g_N != N) circulant_op_init(N);

  const float *W_hc = NULL;
  for (int i = 0; i < g_wreg_n; i++)
    if (g_wreg[i].w == w) { W_hc = g_wreg[i].W_hc; break; }
  if (!W_hc) {
    memcpy(g_w_fallback, w, (size_t)N * sizeof(float));
    execute_rdfft_f32(g_w_fallback, g_W_re, g_W_im, N);
    W_hc = g_w_fallback;
  }

  memcpy(g_scratch, x, (size_t)N * sizeof(float));
  execute_rdfft_f32(g_scratch, g_W_re, g_W_im, N);
  half_complex_mul_inplace(W_hc, g_scratch, N);
  execute_irdfft_f32(g_scratch, g_IW_re, g_IW_im, N);
  memcpy(y, g_scratch, (size_t)N * sizeof(float));
}
