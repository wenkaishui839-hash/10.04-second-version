/* ========================================================================
 * main.c — 任务四第一步: 权重读取 测试主程序
 *
 * 读取两个权重文件, 打印模型结构摘要, 并做两项自检:
 *   1) 层数与每层尺寸是否与 PaddedBlockCirculantNICE 一致;
 *   2) 总参数量是否等于 105472 (= 104448 循环层 + 1024 缩放层)。
 *
 * 用法:
 *   ./load_weights_test [circulant_weights.bin] [scaling_weights.bin]
 * 两个路径缺省为当前目录下的文件名, 也可传入绝对路径。
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>

#include "load_weights.h"

int main(int argc, char **argv) {
    const char *circ_path  = argc > 1 ? argv[1] : "circulant_weights.bin";
    const char *scale_path = argc > 2 ? argv[2] : "scaling_weights.bin";

    NICEModel model;

    if (load_circulant_weights(circ_path, &model) != 0) {
        fprintf(stderr, "读取 circulant_weights 失败\n");
        return 1;
    }
    if (load_scaling_weights(scale_path, &model.scaling) != 0) {
        fprintf(stderr, "读取 scaling_weights 失败\n");
        free_nice_model(&model);
        return 1;
    }

    printf("== NICE 权重读取结果 ==\n");
    printf("block-circulant 层数 : %d\n", model.n_layers);
    printf("ScalingLayer 特征数  : %d\n\n", model.scaling.num_features);

    long total_params = 0;
    double weight_sum = 0.0;

    for (int i = 0; i < model.n_layers; i++) {
        BlockCirculantLayer *L = &model.layers[i];
        int nw = L->n_block_rows * L->n_block_cols * L->block_size;
        printf("[层 %2d] in=%4d out=%4d block=%3d rows=%d cols=%d | w0=% .6f b0=% .6f\n",
               i, L->in_features, L->out_features, L->block_size,
               L->n_block_rows, L->n_block_cols,
               L->weight ? L->weight[0] : 0.0f,
               L->bias   ? L->bias[0]   : 0.0f);
        for (int j = 0; j < nw; j++) weight_sum += (double)L->weight[j];
        for (int j = 0; j < L->out_features; j++) weight_sum += (double)L->bias[j];
        total_params += (long)nw + L->out_features;
    }

    /* 缩放层: log_scale 是可训练参数 (1024 个), scale 是其 exp 派生值。 */
    for (int j = 0; j < model.scaling.num_features; j++) {
        weight_sum += (double)model.scaling.log_scale[j];
        total_params++;
    }

    printf("\nscale[0]=%.6f  scale[512]=%.6f  scale[1023]=%.6f\n",
           model.scaling.scale[0],
           model.scaling.scale[512],
           model.scaling.scale[1023]);
    printf("总参数量            : %ld (期望 105472)\n", total_params);
    printf("权重数值和 (校验用) : %.9g\n", weight_sum);

    if (model.n_layers == 24 && total_params == 105472) {
        printf("\n[自检通过] 层数与参数量与 PaddedBlockCirculantNICE 一致。\n");
    } else {
        printf("\n[自检失败] 层数=%d 参数=%ld, 请检查权重文件或解析代码。\n",
               model.n_layers, total_params);
    }

    free_nice_model(&model);
    return 0;
}
