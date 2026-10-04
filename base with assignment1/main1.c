#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N);

/* 板上计时: 直接发 SYS_clock_gettime(CLOCK_MONOTONIC) (riscv64 __NR=113),
 * 不引入 libc 符号 (与任务四 infer_main.c 的 now_monotonic 同理)。
 * Spike/pk 下计时无意义但无害, 真板子上即为墙上时间。 */
static void now_monotonic(struct timespec *ts) {
  register long r_a0 __asm__("a0") = 1; /* CLOCK_MONOTONIC */
  register long r_a1 __asm__("a1") = (long)ts;
  register long r_a7 __asm__("a7") = 113;
  __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a7) : "memory");
}

int main() {
    /* 关闭 stdout 缓冲，确保 START/END 标记即时输出到管道 */
    setvbuf(stdout, NULL, _IONBF, 0);

    /* === 可调参数 === */
    const int N      = 256;   /* FFT 点数 (2的幂) */
    const int WARMUP = 10;    /* 预热次数 */
    const int REPS   = 100;   /* 正式跑分次数 */

    float* Y    = (float*)malloc(N * sizeof(float));
    float* W_re = (float*)malloc((N / 2) * sizeof(float));
    float* W_im = (float*)malloc((N / 2) * sizeof(float));

    /* 构造测试信号 */
    for (int n = 0; n < N; n++) {
        Y[n] = (float)sin(2.0 * M_PI * n / N)
             + 0.5f * (float)sin(4.0 * M_PI * n / N);
    }

    /* 预生成旋转因子 (不计时) */
    generate_rdfft_twiddle_factors(W_re, W_im, N);

    /* 预热 */
    for (int r = 0; r < WARMUP; r++) {
        execute_rdfft_rvv_f32(Y, W_re, W_im, N);
    }

    /* ---- 从这里开始用 Spike -l 统计指令 ---- */
    struct timespec t0, t1;
    now_monotonic(&t0);
    printf("START\n");
    for (int r = 0; r < REPS; r++) {
        execute_rdfft_rvv_f32(Y, W_re, W_im, N);
    }
    printf("END\n");
    now_monotonic(&t1);
    /* ---- 统计到这里结束 ---- */

    /* 板上计时结果 (Spike 下无意义, 仅真板子有效; 在 END 之后不影响计数区间) */
    {
        double secs = (double)(t1.tv_sec - t0.tv_sec)
                    + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("[time] reps=%d total=%.6fs per_fft=%.3fus\n",
               REPS, secs, secs / REPS * 1e6);
    }

    /* 抽样检查频点 (确认结果正确)。注意: 上面 REPS 次连续 FFT 会把幅度放大
     * sqrt(N)^REPS 倍溢出为 nan (预期现象, 不代表算子错误), 故这里先重置
     * 测试信号再单独做一次 FFT。本段在 END 之后, 不影响计数与计时区间。 */
    for (int n = 0; n < N; n++) {
        Y[n] = (float)sin(2.0 * M_PI * n / N)
             + 0.5f * (float)sin(4.0 * M_PI * n / N);
    }
    execute_rdfft_rvv_f32(Y, W_re, W_im, N);
    /* 期望值: 信号 = sin(2πn/N) + 0.5·sin(4πn/N), 半复数布局 Y[k]=Re(F[k]),
     * Y[N-k]=Im(F[k])。故 Y[0]=0 (直流), Y[1]=Re(F[1])=0,
     * Y[N/2]=0 (奈奎斯特), Y[N-1]=Im(F[1])=-N/2=-128, Y[N-2]=Im(F[2])=-N/4=-64。 */
    printf("Check: Y[0]=%.4f Y[1]=%.4f Y[%d]=%.4f Y[%d]=%.4f Y[%d]=%.4f\n",
           (double)Y[0], (double)Y[1], N/2, (double)Y[N/2],
           N-1, (double)Y[N-1], N-2, (double)Y[N-2]);

    free(Y); free(W_re); free(W_im);
    return 0;
}
