/* ========================================================================
 * circulant_v3.c — 循环卷积算子 (RVV, v3 三段式零 malloc)
 *
 * 接口 (circulant_op.h):
 *   circulant_op_init(N)           加载期一次: 旋转因子表 + 常驻工作缓冲
 *   circulant_op_prepare_weight(w) 加载期: 权重 rdFFT 频谱预计算注册
 *   circulant_op_f32(w, x, y, N)   前向: 零 malloc (查注册表 + 复用缓冲)
 *
 * 满足"前向/反向过程零动态工作区"要求:
 *   - x 的频谱在常驻 g_scratch 上原地计算 (N floats, init 时分配)
 *   - 半复数频域乘为原地乘法 (B = A .* B, 先读后写)
 *   - 权重频谱来自加载期预计算注册表; 未注册权重走常驻 g_w_fallback
 *     兜底 (同样零 malloc)
 * ======================================================================== */
#include <stdlib.h>
#include <string.h>

#include "circulant_op.h"

extern void generate_rdfft_twiddle_factors_staged(float *W_re, float *W_im, int N);
extern void execute_rdfft_v3_f32(float *Y, const float *W_re, const float *W_im, int N);
extern void execute_irdfft_v3_f32(float *Y, const float *W_re, const float *W_im, int N);
extern void half_complex_mul_v3(const float *A, float *B, int N);

/* 常驻状态: init 一次分配, 前向过程不再 malloc */
static int    g_N = 0;
static float *g_W_re = NULL, *g_W_im = NULL;  /* 逐级连续旋转因子表 */
static float *g_scratch = NULL;               /* x 频谱原地工作区 (N) */
static float *g_w_fallback = NULL;            /* 未注册权重兜底频谱 (N) */

/* 权重频谱注册表 (prepare 阶段填充, 指针键) */
typedef struct { const float *w; float *W_hc; } WEntry;
static WEntry *g_wreg = NULL;
static int g_wreg_n = 0, g_wreg_cap = 0;

void circulant_op_init(int N) {
  if (g_N == N && g_W_re) return;
  free(g_W_re); free(g_W_im); free(g_scratch); free(g_w_fallback);
  g_N = N;
  g_W_re  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_W_im  = (float *)malloc((size_t)(N / 2) * sizeof(float));
  g_scratch = (float *)malloc((size_t)N * sizeof(float));
  g_w_fallback = (float *)malloc((size_t)N * sizeof(float));
  generate_rdfft_twiddle_factors_staged(g_W_re, g_W_im, N);
  g_wreg_n = 0;  /* N 变了, 旧频谱作废 */
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
  execute_rdfft_v3_f32(W_hc, g_W_re, g_W_im, g_N);
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
    /* 兜底: 未注册权重在常驻缓冲上现算频谱 (仍零 malloc) */
    memcpy(g_w_fallback, w, (size_t)N * sizeof(float));
    execute_rdfft_v3_f32(g_w_fallback, g_W_re, g_W_im, N);
    W_hc = g_w_fallback;
  }

  /* 1. rdFFT(x) 在 g_scratch 上原地计算 */
  memcpy(g_scratch, x, (size_t)N * sizeof(float));
  execute_rdfft_v3_f32(g_scratch, g_W_re, g_W_im, N);

  /* 2. 半复数频域乘 (原地: scratch = W_hc .* scratch) */
  half_complex_mul_v3(W_hc, g_scratch, N);

  /* 3. irdFFT 原地还原时域结果 */
  execute_irdfft_v3_f32(g_scratch, g_W_re, g_W_im, N);

  memcpy(y, g_scratch, (size_t)N * sizeof(float));
}
