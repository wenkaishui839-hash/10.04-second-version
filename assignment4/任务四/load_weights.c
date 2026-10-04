/* ========================================================================
 * load_weights.c — 任务四第一步: 权重读取 (fopen / fread 二进制)
 *
 * 读取由 convert_weights.py 从任务三文本权重转出的二进制文件:
 *   - circulant_weights.bin : 24 层 block-circulant 全连接层
 *   - scaling_weights.bin   : ScalingLayer 缩放层
 *
 * 二进制格式 (little-endian, 与 RISC-V / x86 主机字节序一致, 直接 fread):
 *
 *   circulant_weights.bin:
 *     uint32 magic = 0x43495243 ("CIRC")
 *     uint32 n_layers
 *     每层重复 n_layers 次:
 *       int32  in_features
 *       int32  out_features
 *       int32  block_size
 *       int32  n_block_rows
 *       int32  n_block_cols
 *       float  weight[n_block_rows * n_block_cols * block_size]
 *       float  bias[out_features]
 *
 *   scaling_weights.bin:
 *     uint32 magic = 0x5343414C ("SCAL")
 *     int32  num_features
 *     float  log_scale[num_features]
 *     float  scale[num_features]
 *
 * 相比原先"整文件读入 + strtod 逐个数解析"的文本方案, 二进制 fread 把一次性
 * 的 ~104448 次 strtod 标量解析降为若干次整块 fread, 大幅减少 Spike 统计到的
 * 标量指令, 使 RVV 跑分的向量化率更接近推理内核的真实水平。数值上 float32
 * 存值与 strtod->(float) 逐位一致, 推理结果与转换前完全相同。
 * ======================================================================== */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "load_weights.h"

#define MAGIC_CIRCULANT 0x43495243u  /* "CIRC" */
#define MAGIC_SCALING   0x5343414Cu  /* "SCAL" */

/* fread 整块读取, 读满 n 字节返回 0, 否则 -1 */
static int read_full(FILE *fp, void *dst, size_t n) {
    return fread(dst, 1, n, fp) == n ? 0 : -1;
}

/* 释放 ScalingLayer 并置空指针 (幂等, 可安全重复调用)。 */
static void scaling_release(ScalingLayer *s) {
    if (!s) return;
    free(s->log_scale);
    free(s->scale);
    s->log_scale = NULL;
    s->scale = NULL;
}

int load_circulant_weights(const char *path, NICEModel *model) {
    if (!model) return -1;
    memset(model, 0, sizeof(*model));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[load_weights] 无法打开文件: %s\n", path);
        return -1;
    }

    uint32_t magic = 0, n_layers = 0;
    if (read_full(fp, &magic, 4) || read_full(fp, &n_layers, 4) ||
        magic != MAGIC_CIRCULANT || n_layers == 0 || n_layers > 1024) {
        fprintf(stderr, "[load_weights] %s 不是有效二进制权重 (magic=0x%08X n=%u)\n",
                path, magic, n_layers);
        fclose(fp);
        return -1;
    }

    model->layers = (BlockCirculantLayer *)calloc((size_t)n_layers,
                                                  sizeof(BlockCirculantLayer));
    if (!model->layers) { fclose(fp); return -1; }
    model->n_layers = (int)n_layers;

    for (uint32_t i = 0; i < n_layers; i++) {
        BlockCirculantLayer *L = &model->layers[i];
        int32_t in_f = 0, out_f = 0, bs = 0, nr = 0, nc = 0;
        if (read_full(fp, &in_f, 4) || read_full(fp, &out_f, 4) ||
            read_full(fp, &bs, 4) || read_full(fp, &nr, 4) || read_full(fp, &nc, 4)) {
            fprintf(stderr, "[load_weights] %s 第 %u 层头部读取失败\n", path, i);
            goto fail;
        }
        L->in_features  = (int)in_f;
        L->out_features = (int)out_f;
        L->block_size   = (int)bs;
        L->n_block_rows = (int)nr;
        L->n_block_cols = (int)nc;

        size_t wcount = (size_t)nr * (size_t)nc * (size_t)bs;
        if (wcount == 0 || L->out_features <= 0) {
            fprintf(stderr, "[load_weights] 第 %u 层尺寸异常 (rows=%d cols=%d block=%d out=%d)\n",
                    i, (int)nr, (int)nc, (int)bs, (int)out_f);
            goto fail;
        }

        L->weight = (float *)malloc(wcount * sizeof(float));
        L->bias   = (float *)malloc((size_t)L->out_features * sizeof(float));
        if (!L->weight || !L->bias) {
            fprintf(stderr, "[load_weights] 内存分配失败\n");
            goto fail;
        }

        if (read_full(fp, L->weight, wcount * sizeof(float)) ||
            read_full(fp, L->bias, (size_t)L->out_features * sizeof(float))) {
            fprintf(stderr, "[load_weights] %s 第 %u 层数据读取不完整\n", path, i);
            goto fail;
        }
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    free_nice_model(model);
    return -1;
}

int load_scaling_weights(const char *path, ScalingLayer *scaling) {
    if (!scaling) return -1;
    memset(scaling, 0, sizeof(*scaling));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[load_weights] 无法打开文件: %s\n", path);
        return -1;
    }

    uint32_t magic = 0;
    int32_t n = 0;
    if (read_full(fp, &magic, 4) || read_full(fp, &n, 4) ||
        magic != MAGIC_SCALING || n <= 0 || n > (1 << 20)) {
        fprintf(stderr, "[load_weights] %s 不是有效二进制缩放权重 (magic=0x%08X n=%d)\n",
                path, magic, n);
        fclose(fp);
        return -1;
    }
    scaling->num_features = (int)n;

    scaling->log_scale = (float *)malloc((size_t)n * sizeof(float));
    scaling->scale     = (float *)malloc((size_t)n * sizeof(float));
    if (!scaling->log_scale || !scaling->scale) {
        fprintf(stderr, "[load_weights] 内存分配失败\n");
        fclose(fp);
        scaling_release(scaling);
        return -1;
    }

    if (read_full(fp, scaling->log_scale, (size_t)n * sizeof(float)) ||
        read_full(fp, scaling->scale, (size_t)n * sizeof(float))) {
        fprintf(stderr, "[load_weights] %s 数据读取不完整\n", path);
        fclose(fp);
        scaling_release(scaling);
        return -1;
    }

    fclose(fp);
    return 0;
}

void free_nice_model(NICEModel *model) {
    if (!model) return;
    for (int i = 0; i < model->n_layers; i++) {
        free(model->layers[i].weight);
        free(model->layers[i].bias);
    }
    free(model->layers);
    scaling_release(&model->scaling);
    memset(model, 0, sizeof(*model));
}
