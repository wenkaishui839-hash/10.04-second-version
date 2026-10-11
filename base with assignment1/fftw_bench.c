/* ========================================================================
 * fftw_bench.c — FFTW 3.3.10（RISC-V 交叉编译, 单精度）vs 本组 rdFFT 同板对比
 *
 * FFTW 无 RVV 后端 → 在 K230 上为纯标量实现。对比口径: 实数 256 点前向 FFT,
 * 同一测试信号 k=1 频点理论真值 (0, -128) 互验。
 *
 * FFTW 构建 (VM, 已于 vmshare 完成 configure/make):
 *   cd <vmshare挂载路径>/fftw-3.3.10 && make install   # 装到 /home/wks/riscv_fftw_install
 *
 * 编译:
 *   riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 \
 *     -o fftw_bench fftw_bench.c external.c twiddle_factors.c \
 *     rdfft_butterfly_compute.c \
 *     -I/home/wks/riscv_fftw_install/include \
 *     -L/home/wks/riscv_fftw_install/lib -lfftw3f -lm
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fftw3.h>

extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
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

static float g_src[N];

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    float* Y    = (float*)malloc(N * sizeof(float));
    float* W_re = (float*)malloc((N/2) * sizeof(float));
    float* W_im = (float*)malloc((N/2) * sizeof(float));
    fftwf_complex* fout = (fftwf_complex*)fftwf_malloc((N/2+1) * sizeof(fftwf_complex));
    if (!Y || !W_re || !W_im || !fout) { fprintf(stderr, "内存分配失败\n"); return 1; }

    /* 测试信号 (与 fft_bench 一致) */
    for (int n = 0; n < N; n++)
        g_src[n] = (float)sin(2.0 * M_PI * n / N) + 0.5f * (float)sin(4.0 * M_PI * n / N);
    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);

    /* FFTW plan (计时区外; ESTIMATE 无运行时调优, MEASURE 板上实测调优) */
    fftwf_plan p_est  = fftwf_plan_dft_r2c_1d(N, g_src, fout, FFTW_ESTIMATE);
    fftwf_plan p_meas = fftwf_plan_dft_r2c_1d(N, g_src, fout, FFTW_MEASURE);

    /* ---- 正确性互验: k=1 频点理论真值 (0, -128) ---- */
    memcpy(Y, g_src, N * sizeof(float));
    execute_rdfft_rvv_f32(Y, W_re, W_im, N);
    printf("[check] 本组 rdFFT (RVV)  : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)Y[1], (double)Y[N-1]);

    fftwf_execute_dft_r2c(p_est, g_src, fout);
    printf("[check] FFTW r2c ESTIMATE : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)fout[1][0], (double)fout[1][1]);

    fftwf_execute_dft_r2c(p_meas, g_src, fout);
    printf("[check] FFTW r2c MEASURE  : k=1 -> (Re=%.6f, Im=%.6f)  期望 (0, -128)\n",
           (double)fout[1][0], (double)fout[1][1]);

    /* ---- 计时 ---- */
    struct timespec t0, t1;
    double secs;

    for (int r = 0; r < WARMUP; r++) { memcpy(Y, g_src, N*sizeof(float));
                                       execute_rdfft_rvv_f32(Y, W_re, W_im, N); }
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) { memcpy(Y, g_src, N*sizeof(float));
                                     execute_rdfft_rvv_f32(Y, W_re, W_im, N); }
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] 本组 rdFFT (RVV)  : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    for (int r = 0; r < WARMUP; r++) fftwf_execute_dft_r2c(p_est, g_src, fout);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) fftwf_execute_dft_r2c(p_est, g_src, fout);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] FFTW r2c ESTIMATE : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    for (int r = 0; r < WARMUP; r++) fftwf_execute_dft_r2c(p_meas, g_src, fout);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) fftwf_execute_dft_r2c(p_meas, g_src, fout);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] FFTW r2c MEASURE  : reps=%d per_fft=%.3fus\n", REPS, secs / REPS * 1e6);

    fftwf_destroy_plan(p_est);
    fftwf_destroy_plan(p_meas);
    free(Y); free(W_re); free(W_im);
    fftwf_free(fout);
    return 0;
}
