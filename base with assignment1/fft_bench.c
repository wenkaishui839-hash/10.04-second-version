/* ========================================================================
 * fft_bench.c — FFT 库同板对比: 本组 rdFFT vs kissfft (BSD-3, kissfft/ 目录)
 *
 * 编译两个版本 (同一份源码, 用 -DUSE_RVV 区分):
 *   RVV 版:  riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 \
 *              -DUSE_RVV -o fft_bench_rvv fft_bench.c kissfft/kiss_fft.c \
 *              kissfft/kiss_fftr.c external.c twiddle_factors.c \
 *              rdfft_butterfly_compute.c -lm
 *   标量版:  同上但 -march=rv64gc、不带 -DUSE_RVV, 并把 rdfft_butterfly_compute.c
 *            换成 pure_clanguage/external.c + rdfft_butterfly_compute_scalar.c
 *
 * 每轮: 预热 + REPS 次, 计时取平均, 输出 [time] 行; 三个实现互相对比
 * 同一测试信号的 k=1 频点 (理论真值 (Re,Im) = (0, -N/2) = (0, -128)):
 *   本组 rdFFT: 半复数布局 Y[1]=Re, Y[N-1]=Im
 *   kiss_fftr:  fout[1].r / fout[1].i (实数专用接口)
 *   kiss_fft:   fout[1].r / fout[1].i (复数接口, 虚部填 0)
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "kissfft/kiss_fft.h"
#include "kissfft/kiss_fftr.h"

#ifdef USE_RVV
extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N);
#else
extern void execute_rdfft_scalar(float* Y, const float* W_re, const float* W_im, int N);
/* 标量孪生 (pure_clanguage) 保持旧 strided 表实现, 用旧生成器 */
extern void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N);
#endif

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

    float *src   = (float*)malloc(N * sizeof(float));   /* 固定测试信号 (不可破坏) */
    float *Y     = (float*)malloc(N * sizeof(float));   /* 本组 rdFFT 原地工作区 */
    float *W_re  = (float*)malloc((N/2) * sizeof(float));
    float *W_im  = (float*)malloc((N/2) * sizeof(float));

    kiss_fft_cpx  *cin  = (kiss_fft_cpx*)malloc(N * sizeof(kiss_fft_cpx));
    kiss_fft_cpx  *cout = (kiss_fft_cpx*)malloc(N * sizeof(kiss_fft_cpx));
    kiss_fft_cpx  *coutr = (kiss_fft_cpx*)malloc((N/2+1) * sizeof(kiss_fft_cpx));

    if (!src || !Y || !W_re || !W_im || !cin || !cout || !coutr) {
        fprintf(stderr, "内存分配失败\n"); return 1;
    }

    /* 测试信号: sin(2πn/N) + 0.5·sin(4πn/N) → k=1 频点理论真值 (0, -128) */
    for (int n = 0; n < N; n++) {
        src[n] = (float)sin(2.0 * M_PI * n / N) + 0.5f * (float)sin(4.0 * M_PI * n / N);
        cin[n].r = src[n]; cin[n].i = 0.0f;
    }
#ifdef USE_RVV
    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);
#else
    generate_rdfft_twiddle_factors(W_re, W_im, N);
#endif

    /* kissfft plan (计时区外) */
    kiss_fft_cfg  cfg  = kiss_fft_alloc(N, 0, NULL, NULL);
    kiss_fftr_cfg cfgr = kiss_fftr_alloc(N, 0, NULL, NULL);

    /* ---- 正确性: 三个实现各跑一次, 打印 k=1 频点 ---- */
    memcpy(Y, src, N * sizeof(float));
#ifdef USE_RVV
    execute_rdfft_rvv_f32(Y, W_re, W_im, N);
#else
    execute_rdfft_scalar(Y, W_re, W_im, N);
#endif
    printf("[check] 本组 rdFFT : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)Y[1], (double)Y[N-1]);

    kiss_fftr(cfgr, src, coutr);
    printf("[check] kiss_fftr   : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)coutr[1].r, (double)coutr[1].i);

    kiss_fft(cfg, cin, cout);
    printf("[check] kiss_fft    : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)cout[1].r, (double)cout[1].i);

    /* ---- 计时 ---- */
    struct timespec t0, t1;
    double secs;

    /* 本组 rdFFT (原地, 每次先恢复输入) */
    for (int r = 0; r < WARMUP; r++) { memcpy(Y, src, N*sizeof(float));
#ifdef USE_RVV
        execute_rdfft_rvv_f32(Y, W_re, W_im, N);
#else
        execute_rdfft_scalar(Y, W_re, W_im, N);
#endif
    }
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) { memcpy(Y, src, N*sizeof(float));
#ifdef USE_RVV
        execute_rdfft_rvv_f32(Y, W_re, W_im, N);
#else
        execute_rdfft_scalar(Y, W_re, W_im, N);
#endif
    }
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] 本组 rdFFT : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    /* kiss_fftr (实数专用, 输出不破坏输入) */
    for (int r = 0; r < WARMUP; r++) kiss_fftr(cfgr, src, coutr);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) kiss_fftr(cfgr, src, coutr);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] kiss_fftr   : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    /* kiss_fft (复数接口) */
    for (int r = 0; r < WARMUP; r++) kiss_fft(cfg, cin, cout);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) kiss_fft(cfg, cin, cout);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] kiss_fft    : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    free(src); free(Y); free(W_re); free(W_im);
    free(cin); free(cout); free(coutr);
    return 0;
}
