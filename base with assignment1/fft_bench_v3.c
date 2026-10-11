/* ========================================================================
 * fft_bench_v3.c — v3 codelet 引擎算子级跑分 (rdFFT(256), 板上计时)
 *
 * 与 fft_bench.c 完全同口径 (同一测试信号 / WARMUP / REPS / [check] 频点),
 * 唯一区别: 调用 v3 引擎 execute_rdfft_v3_f32, 用于与归档的 v2 数据
 * (fft_bench_rvv = 10.74µs) 直接对比算子级加速。
 *
 * 编译 (VM, 任务一目录):
 *   riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 \
 *     -o fft_bench_v3_rvv fft_bench_v3.c external_v3.c twiddle_factors.c \
 *     rdfft_butterfly_compute.c rdfft_butterfly_v3.c -lm
 * 板上: taskset -c 0 ./fft_bench_v3_rvv
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern void execute_rdfft_v3_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N);

/* 板上计时: 原始 SYS_clock_gettime (riscv64 __NR=113), 不引入 libc 符号 */
static void now_monotonic(struct timespec *ts) {
  register long r_a0 __asm__("a0") = 1; /* CLOCK_MONOTONIC */
  register long r_a1 __asm__("a1") = (long)ts;
  register long r_a7 __asm__("a7") = 113;
  __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a7) : "memory");
}

#define N       256
#define WARMUP  20
#define REPS    200

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    float *src  = (float*)malloc(N * sizeof(float));
    float *Y    = (float*)malloc(N * sizeof(float));
    float *W_re = (float*)malloc((N/2) * sizeof(float));
    float *W_im = (float*)malloc((N/2) * sizeof(float));
    if (!src || !Y || !W_re || !W_im) {
        fprintf(stderr, "内存分配失败\n"); return 1;
    }

    for (int n = 0; n < N; n++)
        src[n] = (float)sin(2.0 * M_PI * n / N) + 0.5f * (float)sin(4.0 * M_PI * n / N);

    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);

    /* 正确性: k=1 频点理论真值 (0, -128) */
    memcpy(Y, src, N * sizeof(float));
    execute_rdfft_v3_f32(Y, W_re, W_im, N);
    printf("[check] v3 rdFFT : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)Y[1], (double)Y[N-1]);

    struct timespec t0, t1;
    for (int r = 0; r < WARMUP; r++) {
        memcpy(Y, src, N * sizeof(float));
        execute_rdfft_v3_f32(Y, W_re, W_im, N);
    }
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) {
        memcpy(Y, src, N * sizeof(float));
        execute_rdfft_v3_f32(Y, W_re, W_im, N);
    }
    now_monotonic(&t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] v3 rdFFT : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    free(src); free(Y); free(W_re); free(W_im);
    return 0;
}
