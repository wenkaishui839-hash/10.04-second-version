/* ========================================================================
 * inference.h — NICE 生成方向 (Z -> X) 前向推理
 *
 * 针对任务三的 PaddedBlockCirculantNICE (padded-block-circ) 模型:
 *   输入 1024 维噪声 z, 输出 784 维 MNIST 样本 (28x28 展平)。
 * ======================================================================== */
#ifndef INFERENCE_H
#define INFERENCE_H

#include "load_weights.h"

#define NICE_PADDED    1024   /* 补零后的特征维 (= scaling 维度) */
#define NICE_HALF       512   /* 切分后的一半 */
#define NICE_ORIGINAL   784   /* MNIST 28x28 展平 */
#define NICE_BLOCK      256   /* block-circulant 块大小 (= FFT 长度) */

/* work 缓冲区大小: 3*1024 (x/a/b) + 512 (net_out) + 2*256 (acc/tmp) = 4096 */
#define NICE_WORK_FLOATS 4096

/* 生成方向前向推理:
 *   z      输入噪声, 长度 NICE_PADDED (1024)
 *   x_out  输出样本, 长度 NICE_ORIGINAL (784)
 *   work   工作缓冲区, 长度 >= NICE_WORK_FLOATS
 *
 * 流程: x = z/scale -> 逆序 coupling block 3..0 -> 裁剪 1024->784。 */
void nice_sample(const NICEModel *model, const float *z, float *x_out, float *work);

#endif /* INFERENCE_H */
