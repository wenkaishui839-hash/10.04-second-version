/* ========================================================================
 * inference_scalar.c — NICE 前向推理主逻辑 (纯标量孪生文件)
 *
 * 与 inference.c (手工 RVV 向量化版) 数学完全一致, 仅实现方式不同:
 * 本文件全部用标量循环, 供 bench.sh 编译纯标量基线 (-march=rv64gc,
 * 无 V 扩展, 保证编译器不会自动向量化), 用于速度对比。
 *
 * 复现 PaddedBlockCirculantNICE.sample():
 *   1. ScalingLayer.inverse:  x = z / scale        (scale = exp(log_scale))
 *   2. 逆序 coupling block 3 -> 0, 每块做一次可逆耦合:
 *        h1, h2 = 前半 512, 后半 512
 *        reverse 时 h1 -= net(h2), 否则 h2 -= net(h1)
 *   3. 裁剪 1024 -> 784
 *
 * 每个 block 的 net(h) 是 6 层 block-circulant 全连接 + ReLU:
 *   BCL(512->1024) -> ReLU -> [BCL(1024->1024) -> ReLU] x4 -> BCL(1024->512)
 * BCL 的卷积部分调用 circulant_op_f32 (N=256), 即任务一的循环卷积算子。
 * ======================================================================== */
#include <stdlib.h>
#include <string.h>

#include "inference.h"
#include "circulant_op.h"

static void relu(float *x, int n) {
  for (int i = 0; i < n; i++)
    if (x[i] < 0.0f) x[i] = 0.0f;
}

/* 单个 block-circulant 全连接层: y = W @ x + b。
 *   y_block[r] = sum_c circ_conv_256(w[r][c], x_block[c]) + bias_block[r]
 * acc/tmp 为长度 block_size 的工作缓冲区 (由调用者传入, 避免反复分配)。 */
static void block_circulant_linear(const BlockCirculantLayer *L,
                                   const float *x, float *y,
                                   float *acc, float *tmp) {
  const int bs = L->block_size;
  for (int r = 0; r < L->n_block_rows; r++) {
    memset(acc, 0, (size_t)bs * sizeof(float));
    for (int c = 0; c < L->n_block_cols; c++) {
      const float *w  = L->weight + ((size_t)r * L->n_block_cols + c) * bs;
      const float *xb = x + (size_t)c * bs;
      circulant_op_f32(w, xb, tmp, bs);
      for (int k = 0; k < bs; k++) acc[k] += tmp[k];
    }
    const float *b = L->bias + (size_t)r * bs;
    float *yb = y + (size_t)r * bs;
    for (int k = 0; k < bs; k++) yb[k] = acc[k] + b[k];
  }
}

/* 一个 block 内的耦合小网络 net(h): 512 -> 1024 -> ... -> 512。
 * a/b 为两个 1024 缓冲 (乒乓避免原地覆盖), out 为 512 缓冲。 */
static void coupling_net(const NICEModel *model, int block_idx,
                         const float *h, float *out,
                         float *a, float *b, float *acc, float *tmp) {
  const BlockCirculantLayer *L = model->layers + block_idx * 6;
  const int HID = NICE_PADDED;   /* 1024 */

  /* 层 0: 512 -> 1024 */
  block_circulant_linear(&L[0], h, a, acc, tmp);
  relu(a, HID);

  /* 层 1..4: 1024 -> 1024 */
  float *in = a, *outbuf = b;
  for (int i = 1; i <= 4; i++) {
    block_circulant_linear(&L[i], in, outbuf, acc, tmp);
    relu(outbuf, HID);
    float *t = in; in = outbuf; outbuf = t;
  }

  /* 层 5: 1024 -> 512 */
  block_circulant_linear(&L[5], in, out, acc, tmp);
}

/* 单个 coupling block 的可逆变换 (inverse, op=sub)。 */
static void coupling_block_inverse(const NICEModel *model, int block_idx, int reverse,
                                   float *x, float *net_out,
                                   float *a, float *b, float *acc, float *tmp) {
  float *h1 = x;
  float *h2 = x + NICE_HALF;
  if (reverse) {
    coupling_net(model, block_idx, h2, net_out, a, b, acc, tmp);
    for (int i = 0; i < NICE_HALF; i++) h1[i] -= net_out[i];
  } else {
    coupling_net(model, block_idx, h1, net_out, a, b, acc, tmp);
    for (int i = 0; i < NICE_HALF; i++) h2[i] -= net_out[i];
  }
}

void nice_sample(const NICEModel *model, const float *z, float *x_out, float *work) {
  /* work 布局: x(1024) | a(1024) | b(1024) | net_out(512) | acc(256) | tmp(256) */
  float *x       = work;
  float *a       = work + NICE_PADDED;                      /* 1024 */
  float *b       = work + 2 * NICE_PADDED;                  /* 2048 */
  float *net_out = work + 3 * NICE_PADDED;                  /* 3072 */
  float *acc     = work + 3 * NICE_PADDED + NICE_HALF;      /* 3584 */
  float *tmp     = acc + NICE_BLOCK;                        /* 3840 */

  /* 1. ScalingLayer.inverse: x = z / scale */
  for (int i = 0; i < NICE_PADDED; i++)
    x[i] = z[i] / model->scaling.scale[i];

  /* 2. 逆序 coupling block 3 -> 0; reverse 标志 {0,1,0,1} */
  static const int reverse_flag[4] = {0, 1, 0, 1};
  for (int bi = 3; bi >= 0; bi--)
    coupling_block_inverse(model, bi, reverse_flag[bi], x, net_out, a, b, acc, tmp);

  /* 3. 裁剪 1024 -> 784 */
  memcpy(x_out, x, NICE_ORIGINAL * sizeof(float));
}
