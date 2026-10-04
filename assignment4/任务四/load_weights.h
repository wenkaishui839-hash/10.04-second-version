/* ========================================================================
 * load_weights.h — 任务四: 权重读取 头文件
 *
 * 定义 NICE 模型在 C 内存中的数据结构, 并声明从任务三导出的
 * 两个权重文件读入内存的函数 (二进制, 由 convert_weights.py 从文本转出):
 *   - circulant_weights.bin : 24 层 block-circulant 全连接层
 *   - scaling_weights.bin   : ScalingLayer 缩放层
 *
 * 数据布局与任务三 PaddedBlockCirculantNICE 的生成方向 (Z -> X) 对齐,
 * 供第二步手写前向推理主逻辑直接消费。
 * ======================================================================== */
#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* 单个 block-circulant 全连接层 (block_size = 256)。
 *
 * 前向计算等价于: 把输入按 block_size 切成 n_block_cols 块, 对每个
 * 输出块 r, y_block[r] = Σ_col  circ_conv_256(weight[r][col], x_block[col]),
 * 最后加上 bias。weight 扁平存放, 长度为 n_block_rows * n_block_cols * block_size。
 */
typedef struct {
    int in_features;    /* 输入维度 (512 / 1024)                          */
    int out_features;   /* 输出维度 (1024 / 512)                         */
    int block_size;     /* 每块大小 = 256 (2 的幂, 直接对应 FFT 长度)     */
    int n_block_rows;   /* 输出方向块数 = ceil(out_features / block_size) */
    int n_block_cols;   /* 输入方向块数 = ceil(in_features / block_size)  */
    float *weight;      /* 长度 n_block_rows * n_block_cols * block_size  */
    float *bias;        /* 长度 out_features                              */
} BlockCirculantLayer;

/* ScalingLayer: 对角线缩放 S = exp(log_scale)。
 *
 * 生成方向的第一步是 x = scaling.inverse(z) = z * exp(-log_scale) = z / scale,
 * 因此推理只用得到 scale; log_scale 一并读入便于核对与调试。
 */
typedef struct {
    int num_features;   /* = 1024                     */
    float *log_scale;   /* 长度 num_features          */
    float *scale;       /* 长度 num_features = exp(log_scale) */
} ScalingLayer;

/* 完整 NICE 模型权重。 */
typedef struct {
    int n_layers;                 /* = 24                                       */
    BlockCirculantLayer *layers;  /* 数组, 顺序与导出顺序一致 (net.0 .. net.3) */
    ScalingLayer scaling;         /* 缩放层权重                                  */
} NICEModel;

/* 从 circulant_weights.bin 读入 24 层 block-circulant 权重。
 * 成功返回 0, 失败返回 -1 (并打印原因到 stderr)。 */
int load_circulant_weights(const char *path, NICEModel *model);

/* 从 scaling_weights.bin 读入缩放层权重。成功返回 0, 失败返回 -1。 */
int load_scaling_weights(const char *path, ScalingLayer *scaling);

/* 释放 load_* 分配的全部内存, 并把 model 清零。 */
void free_nice_model(NICEModel *model);

#ifdef __cplusplus
}
#endif

#endif /* LOAD_WEIGHTS_H */
