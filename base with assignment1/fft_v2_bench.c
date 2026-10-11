/* ========================================================================
 * fft_v2_bench.c — rdFFT v2 结构内优化对比 (原论文共轭对结构不变)
 *
 * 同板对比四种配置 (同一测试信号, 频点理论真值兜底):
 *   v1      : 现有实现 (external.c + rdfft_butterfly_compute.c, j 增量位反转)
 *   v2-mode0: 逐级连续旋转因子表 + j 增量位反转 (只隔离旋转因子收益)
 *   v2-mode1: 逐级表 + 置换表查表交换位反转
 *   v2-mode2: 逐级表 + RVV 双缓冲位反转 (vluxei32)
 *
 * 全部保持: 共轭对蝶形公式 / 隐式半复数布局 / DIT 顺序 / 蝶形真原位。
 *
 * 编译: riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 \
 *         -o fft_v2_bench fft_v2_bench.c external.c twiddle_factors.c \
 *         rdfft_butterfly_compute.c external_staged.c \
 *         rdfft_butterfly_staged.c bitrev_rvv.c -lm
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>

extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void execute_rdfft_staged_f32(float* Y, const float* W_re, const float* W_im,
                                     int N, int bitrev_mode,
                                     float* scratch, const uint32_t* rev);
extern void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N);
extern void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N);
/* 注: v2 已合并入主实现 (execute_rdfft_rvv_f32 现为逐级表+查表位反转),
 * 故本 bench 的 W1 也改用逐级表; v1 行现等价于 v2 mode1。 */
extern void bitrev_table_build(uint32_t* rev, int N);

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

/* 重置测试信号 (sin(2πn/N) + 0.5·sin(4πn/N)) */
static void reset_signal(float* Y) {
    for (int n = 0; n < N; n++) {
        Y[n] = (float)sin(2.0 * M_PI * n / N) + 0.5f * (float)sin(4.0 * M_PI * n / N);
    }
}

/* 频点检查: Y[0]=0, Y[1]=0, Y[N/2]=0, Y[N-1]=-128, Y[N-2]=-64 */
static void check_and_print(const char* name, const float* Y) {
    printf("[check] %-28s: Y[0]=%.4f Y[1]=%.4f Y[%d]=%.4f Y[%d]=%.4f Y[%d]=%.4f  (期望 0,0,0,-128,-64)\n",
           name, (double)Y[0], (double)Y[1], N/2, (double)Y[N/2],
           N-1, (double)Y[N-1], N-2, (double)Y[N-2]);
}

/* 计时并返回 per_fft (µs); 每次调用前重置信号 */
static double bench_run(void (*fn)(void*), void* ctx) {
    for (int r = 0; r < WARMUP; r++) fn(ctx);
    struct timespec t0, t1;
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) fn(ctx);
    now_monotonic(&t1);
    double secs = (double)(t1.tv_sec - t0.tv_sec)
                + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return secs / REPS * 1e6;
}

/* 各配置的上下文与执行函数 */
typedef struct { float* Y; float* W_re; float* W_im; int n; int mode; float* scratch; const uint32_t* rev; } Ctx;

/* 预存输入信号, 计时循环内只用 memcpy 恢复 (sin 极慢, 不能进计时区间) */
static float g_src[N];

static void run_v1(void* p) {
    Ctx* c = (Ctx*)p;
    memcpy(c->Y, g_src, sizeof(g_src));
    execute_rdfft_rvv_f32(c->Y, c->W_re, c->W_im, c->n);
}
static void run_v2(void* p) {
    Ctx* c = (Ctx*)p;
    memcpy(c->Y, g_src, sizeof(g_src));
    execute_rdfft_staged_f32(c->Y, c->W_re, c->W_im, c->n, c->mode, c->scratch, c->rev);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    float*    Y      = (float*)malloc(N * sizeof(float));
    float*    scratch = (float*)malloc(N * sizeof(float));
    float*    W1_re  = (float*)malloc((N/2) * sizeof(float));
    float*    W1_im  = (float*)malloc((N/2) * sizeof(float));
    float*    W2_re  = (float*)malloc((N/2) * sizeof(float));
    float*    W2_im  = (float*)malloc((N/2) * sizeof(float));
    uint32_t* rev    = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!Y || !scratch || !W1_re || !W1_im || !W2_re || !W2_im || !rev) {
        fprintf(stderr, "内存分配失败\n"); return 1;
    }

    generate_rdfft_twiddle_factors_staged(W1_re, W1_im, N);
    generate_rdfft_twiddle_factors_staged(W2_re, W2_im, N);
    bitrev_table_build(rev, N);
    reset_signal(g_src);   /* 预存输入, 计时循环内 memcpy 恢复 */

    /* ---- 正确性: 三种配置各跑一次 (mode2 的 vluxei32 在 C908/clang 上
     * 触发非对齐访存异常, 与早前 GCC/Spike 的 NaN 同源, 已剔除) ---- */
    reset_signal(Y);
    execute_rdfft_rvv_f32(Y, W1_re, W1_im, N);
    check_and_print("v1 现有实现", Y);

    for (int m = 0; m <= 1; m++) {
        reset_signal(Y);
        execute_rdfft_staged_f32(Y, W2_re, W2_im, N, m, scratch, rev);
        char name[64];
        snprintf(name, sizeof(name), "v2 mode%d", m);
        check_and_print(name, Y);
    }

    /* ---- 计时 ---- */
    Ctx c1 = { Y, W1_re, W1_im, N, 0, scratch, rev };
    printf("[time] v1 现有实现          : per_fft=%.3fus\n", bench_run(run_v1, &c1));
    for (int m = 0; m <= 1; m++) {
        Ctx c2 = { Y, W2_re, W2_im, N, m, scratch, rev };
        printf("[time] v2 mode%d (逐级表)   : per_fft=%.3fus\n", m, bench_run(run_v2, &c2));
    }

    free(Y); free(scratch); free(W1_re); free(W1_im);
    free(W2_re); free(W2_im); free(rev);
    return 0;
}
