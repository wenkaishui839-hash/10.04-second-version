/* ========================================================================
 * circulant_op.h — 循环卷积算子统一接口
 *
 * y = C(w) * x, 其中 C(w) 是第一列为 w[0..N-1] 的循环矩阵, 等价于
 * w 与 x 的循环卷积:
 *     y[n] = sum_k w[k] * x[(n - k) mod N]
 * N 必须是 2 的幂 (本模型中恒为 256)。
 *
 * 第二步 (主机校验) 用 circulant_scalar.c 的标量 FFT 实现;
 * 第三步 (Spike 跑分) 换成任务一的 RVV 实现 circulant_matmul_rvv_f32,
 * 接口不变, 推理主逻辑无需改动。
 * ======================================================================== */
#ifndef CIRCULANT_OP_H
#define CIRCULANT_OP_H

#ifdef __cplusplus
extern "C" {
#endif

/* 预计算 N 点旋转因子表 (调用一次即可)。 */
void circulant_op_init(int N);

/* 循环卷积算子: y = C(w) * x。 */
void circulant_op_f32(const float *w, const float *x, float *y, int N);

#ifdef __cplusplus
}
#endif

#endif /* CIRCULANT_OP_H */
