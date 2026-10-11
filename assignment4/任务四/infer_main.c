/* ========================================================================
 * infer_main.c — 任务四第二步: NICE 前向推理 入口
 *
 * 用法:
 *   ./nice_infer [circulant_weights.txt] [scaling_weights.txt] [z_input.txt]
 * 三个参数依次为: 循环层权重 / 缩放层权重 / 输入噪声。
 * 结果写到 stdout, 由 Makefile 重定向到 x_out.txt / x_out_rvv.txt。
 * 缺省为当前目录下的同名文件。
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "load_weights.h"
#include "circulant_op.h"
#include "inference.h"

static int read_float_file(const char *path, float *dst, int n) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "[infer] 无法打开输入文件: %s\n", path);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    if (fscanf(fp, "%f", &dst[i]) != 1) {
      fprintf(stderr, "[infer] 读取 %s 第 %d 个数失败\n", path, i);
      fclose(fp);
      return -1;
    }
  }
  fclose(fp);
  return 0;
}

/* 结果写到 stdout: pk(proxy kernel) 下无法 fopen("w") 创建文件, 且 pk 会把
 * stdout/stderr 都经 HTIF 合并进 spike 的 stdout; 因此这里只往 stdout 写纯数值,
 * 由 Makefile 用 shell 重定向落盘 (x_out.txt / x_out_rvv.txt)。 */
static void emit_float_output(const float *src, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.9g", src[i]);
    if (i + 1 < n) printf(" ");
  }
  printf("\n");
  fflush(stdout);
}

/* 取当前堆顶: 直接发 SYS_brk(0) 系统调用 (riscv64 __NR_brk=214), 不引入 libc
 * 的 sbrk 符号, 避免改变静态 ELF 布局 (pk 加载器对段布局敏感, 曾导致加载失败)。 */
static size_t heap_brk_bytes(void) {
  register long r_a0 __asm__("a0") = 0;
  register long r_a7 __asm__("a7") = 214;
  __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a7) : "memory");
  return (size_t)r_a0;
}

/* 取单调时钟: 直接发 SYS_clock_gettime(CLOCK_MONOTONIC) (riscv64 __NR=113),
 * 与 heap_brk_bytes 同理不引入 libc 符号。Spike/pk 下计时无意义但无害,
 * 真板子上即为墙上时间 (板上计时用)。 */
static void now_monotonic(struct timespec *ts) {
  register long r_a0 __asm__("a0") = 1; /* CLOCK_MONOTONIC */
  register long r_a1 __asm__("a1") = (long)ts;
  register long r_a7 __asm__("a7") = 113;
  __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a7) : "memory");
}

int main(int argc, char **argv) {
  const char *circ_path  = argc > 1 ? argv[1] : "circulant_weights.txt";
  const char *scale_path = argc > 2 ? argv[2] : "scaling_weights.txt";
  const char *z_path     = argc > 3 ? argv[3] : "z_input.txt";

  /* 可选第 4 参数: 推理重复次数 (板上计时用; Spike 跑分不传, 默认 1 次) */
  int reps = (argc > 4) ? atoi(argv[4]) : 1;
  if (reps < 1) reps = 1;

  NICEModel model;
  if (load_circulant_weights(circ_path, &model) != 0) return 1;
  if (load_scaling_weights(scale_path, &model.scaling) != 0) {
    free_nice_model(&model);
    return 1;
  }

  /* 结构校验: 与 padded-block-circ 模型一致 */
  if (model.n_layers != 24 ||
      model.scaling.num_features != NICE_PADDED ||
      model.layers[0].block_size != NICE_BLOCK) {
    fprintf(stderr, "[infer] 模型结构与预期不符: layers=%d num_features=%d block=%d\n",
            model.n_layers, model.scaling.num_features, model.layers[0].block_size);
    free_nice_model(&model);
    return 1;
  }

  float *z     = (float *)malloc(NICE_PADDED * sizeof(float));
  float *x_out = (float *)malloc(NICE_ORIGINAL * sizeof(float));
  float *work  = (float *)malloc(NICE_WORK_FLOATS * sizeof(float));
  if (!z || !x_out || !work) {
    fprintf(stderr, "[infer] 内存分配失败\n");
    return 1;
  }

  if (read_float_file(z_path, z, NICE_PADDED) != 0) return 1;

  circulant_op_init(NICE_BLOCK);

  /* v3: 加载期预计算全部权重频谱并注册 (前向推理过程零 malloc) */
  for (int i = 0; i < model.n_layers; i++) {
    const BlockCirculantLayer *L = &model.layers[i];
    int nblocks = L->n_block_rows * L->n_block_cols;
    for (int j = 0; j < nblocks; j++)
      circulant_op_prepare_weight(L->weight + (size_t)j * L->block_size);
  }

  /* 预热 1 次: 板上避开冷缓存/首次缺页混入计时 (在计数区间外, 不影响跑分) */
  nice_sample(&model, z, x_out, work);

  /* 堆内存测量: brk(0) 取当前堆顶。打印在计数区间之外, 不影响跑分。 */
  fprintf(stderr, "[heap] before_infer=%zu bytes\n", heap_brk_bytes());

  struct timespec t0, t1;
  now_monotonic(&t0);
  fprintf(stderr, "START\n");   // <--- 核心推理开始标记
  for (int r = 0; r < reps; r++)
    nice_sample(&model, z, x_out, work);
  fprintf(stderr, "END\n");     // <--- 核心推理结束标记
  now_monotonic(&t1);

  /* brk 只增不减, 推理后的堆顶即峰值堆占用 (含推理期间的瞬时缓冲区)。 */
  fprintf(stderr, "[heap] after_infer=%zu bytes\n", heap_brk_bytes());

  double secs = (double)(t1.tv_sec - t0.tv_sec)
              + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
  fprintf(stderr, "[time] reps=%d total=%.6fs per_infer=%.1fus\n",
          reps, secs, secs / reps * 1e6);

  emit_float_output(x_out, NICE_ORIGINAL);

  free(z);
  free(x_out);
  free(work);
  free_nice_model(&model);
  return 0;
}
