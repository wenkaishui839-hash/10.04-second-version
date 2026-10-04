/* ========================================================================
 * inference.c — NICE 生成方向前向推理主逻辑 (纯血 RVV 极致优化版)
 * ======================================================================== */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <riscv_vector.h>

#include "inference.h"
#include "circulant_op.h"

/* ---------------------------------------------------------
 * RVV 向量化基础算子区
 * --------------------------------------------------------- */
/* 1. 向量化 ReLU: max(x, 0) */
static void relu_rvv(float *x, int n) {
    size_t vl;
    for (; n > 0; n -= vl, x += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x, vl);
        vfloat32m8_t vzero = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        vx = __riscv_vfmax_vv_f32m8(vx, vzero, vl);
        __riscv_vse32_v_f32m8(x, vx, vl);
    }
}

/* 2. 向量化加法: dst[i] += src[i] */
static void add_array_rvv(float *dst, const float *src, int n) {
    size_t vl;
    for (; n > 0; n -= vl, dst += vl, src += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(dst, vl);
        vfloat32m8_t vs = __riscv_vle32_v_f32m8(src, vl);
        vd = __riscv_vfadd_vv_f32m8(vd, vs, vl);
        __riscv_vse32_v_f32m8(dst, vd, vl);
    }
}

/* 3. 向量化加法并赋值: dst[i] = src1[i] + src2[i] */
static void add_arrays_to_rvv(float *dst, const float *src1, const float *src2, int n) {
    size_t vl;
    for (; n > 0; n -= vl, dst += vl, src1 += vl, src2 += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(src1, vl);
        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(src2, vl);
        vfloat32m8_t vd = __riscv_vfadd_vv_f32m8(v1, v2, vl);
        __riscv_vse32_v_f32m8(dst, vd, vl);
    }
}

/* 4. 向量化减法: dst[i] -= src[i] (用于残差连接) */
static void sub_array_rvv(float *dst, const float *src, int n) {
    size_t vl;
    for (; n > 0; n -= vl, dst += vl, src += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t vd = __riscv_vle32_v_f32m8(dst, vl);
        vfloat32m8_t vs = __riscv_vle32_v_f32m8(src, vl);
        vd = __riscv_vfsub_vv_f32m8(vd, vs, vl);
        __riscv_vse32_v_f32m8(dst, vd, vl);
    }
}

/* 5. 向量化除法: dst[i] = src1[i] / src2[i] (用于 Scaling 层) */
static void div_arrays_to_rvv(float *dst, const float *src1, const float *src2, int n) {
    size_t vl;
    for (; n > 0; n -= vl, dst += vl, src1 += vl, src2 += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vfloat32m8_t v1 = __riscv_vle32_v_f32m8(src1, vl);
        vfloat32m8_t v2 = __riscv_vle32_v_f32m8(src2, vl);
        vfloat32m8_t vd = __riscv_vfdiv_vv_f32m8(v1, v2, vl);
        __riscv_vse32_v_f32m8(dst, vd, vl);
    }
}

/* ---------------------------------------------------------
 * 模型网络构建区
 * --------------------------------------------------------- */
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
      add_array_rvv(acc, tmp, bs);
    }
    const float *b = L->bias + (size_t)r * bs;
    float *yb = y + (size_t)r * bs;
    add_arrays_to_rvv(yb, acc, b, bs);
  }
}

static void coupling_net(const NICEModel *model, int block_idx,
                         const float *h, float *out,
                         float *a, float *b, float *acc, float *tmp) {
  const BlockCirculantLayer *L = model->layers + block_idx * 6;
  const int HID = NICE_PADDED;

  block_circulant_linear(&L[0], h, a, acc, tmp);
  relu_rvv(a, HID);

  float *in = a, *outbuf = b;
  for (int i = 1; i <= 4; i++) {
    block_circulant_linear(&L[i], in, outbuf, acc, tmp);
    relu_rvv(outbuf, HID);
    float *t = in; in = outbuf; outbuf = t;
  }
  block_circulant_linear(&L[5], in, out, acc, tmp);
}

static void coupling_block_inverse(const NICEModel *model, int block_idx, int reverse,
                                   float *x, float *net_out,
                                   float *a, float *b, float *acc, float *tmp) {
  float *h1 = x;
  float *h2 = x + NICE_HALF;
  if (reverse) {
    coupling_net(model, block_idx, h2, net_out, a, b, acc, tmp);
    sub_array_rvv(h1, net_out, NICE_HALF);
  } else {
    coupling_net(model, block_idx, h1, net_out, a, b, acc, tmp);
    sub_array_rvv(h2, net_out, NICE_HALF);
  }
}

void nice_sample(const NICEModel *model, const float *z, float *x_out, float *work) {
  float *x       = work;
  float *a       = work + NICE_PADDED;
  float *b       = work + 2 * NICE_PADDED;
  float *net_out = work + 3 * NICE_PADDED;
  float *acc     = work + 3 * NICE_PADDED + NICE_HALF;
  float *tmp     = acc + NICE_BLOCK;

  div_arrays_to_rvv(x, z, model->scaling.scale, NICE_PADDED);

  static const int reverse_flag[4] = {0, 1, 0, 1};
  for (int bi = 3; bi >= 0; bi--)
    coupling_block_inverse(model, bi, reverse_flag[bi], x, net_out, a, b, acc, tmp);

  memcpy(x_out, x, NICE_ORIGINAL * sizeof(float));
}
