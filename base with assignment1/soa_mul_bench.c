/* ========================================================================
 * soa_mul_bench.c — Memory Layout 微基准实验 (任务: "memory layout 怎么设计最合理")
 *
 * 对比两种布局下的半复数频域逐点乘 (N=256, 循环卷积的频域步):
 *   路径 1 (现状): 镜像半复数布局, half_complex_mul_rvv 直接乘 (含 vrgather 调头)
 *   路径 2 (实验): 先 unpack 成 SoA 双数组 (Re/Im 各 N/2+1), 全连续向量乘,
 *                  再 repack 回镜像布局
 * 两路径结果应逐位一致 (相同运算顺序), 耗时对比回答"换布局是否更快"。
 *
 * 编译: riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 \
 *         -o soa_mul_bench soa_mul_bench.c irdfft_half_complex_mul.c -lm
 * ======================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <riscv_vector.h>

extern void half_complex_mul_rvv(const float* restrict A, const float* restrict B,
                                 float* restrict C, int N);

/* 板上计时: 原始 SYS_clock_gettime (riscv64 __NR=113), 不引入 libc 符号 */
static void now_monotonic(struct timespec *ts) {
  register long r_a0 __asm__("a0") = 1; /* CLOCK_MONOTONIC */
  register long r_a1 __asm__("a1") = (long)ts;
  register long r_a7 __asm__("a7") = 113;
  __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a1), "r"(r_a7) : "memory");
}

#define N       256
#define HALF    (N/2)
#define REPS    2000
#define WARMUP  100

/* 镜像半复数 -> SoA: ReA[0..HALF]=A[k]; ImA[k]=A[N-k] (1..HALF-1), 两端置 0 */
static void unpack_soa(const float* A, float* ReA, float* ImA) {
  int k = 0;
  while (k <= HALF) {
    size_t vl = __riscv_vsetvl_e32m1(HALF - k + 1);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(&A[k], vl);
    __riscv_vse32_v_f32m1(&ReA[k], v, vl);
    k += (int)vl;
  }
  ImA[0] = 0.0f; ImA[HALF] = 0.0f;
  k = 1;
  while (k < HALF) {
    size_t vl = __riscv_vsetvl_e32m1(HALF - k);
    vuint32m1_t idx = __riscv_vid_v_u32m1(vl);
    vuint32m1_t rev = __riscv_vrsub_vx_u32m1(idx, vl - 1, vl);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(&A[N - k - (int)vl + 1], vl);
    v = __riscv_vrgather_vv_f32m1(v, rev, vl);
    __riscv_vse32_v_f32m1(&ImA[k], v, vl);
    k += (int)vl;
  }
}

/* SoA 全连续复数乘: ReC=ReA*ReB-ImA*ImB, ImC=ReA*ImB+ImA*ReB (无 vrgather) */
static void mul_soa(const float* ReA, const float* ImA,
                    const float* ReB, const float* ImB,
                    float* ReC, float* ImC) {
  int k = 0;
  while (k <= HALF) {
    size_t vl = __riscv_vsetvl_e32m1(HALF - k + 1);
    vfloat32m1_t vReA = __riscv_vle32_v_f32m1(&ReA[k], vl);
    vfloat32m1_t vImA = __riscv_vle32_v_f32m1(&ImA[k], vl);
    vfloat32m1_t vReB = __riscv_vle32_v_f32m1(&ReB[k], vl);
    vfloat32m1_t vImB = __riscv_vle32_v_f32m1(&ImB[k], vl);
    vfloat32m1_t vReC = __riscv_vfmul_vv_f32m1(vReA, vReB, vl);
    vReC = __riscv_vfnmsac_vv_f32m1(vReC, vImA, vImB, vl);
    vfloat32m1_t vImC = __riscv_vfmul_vv_f32m1(vReA, vImB, vl);
    vImC = __riscv_vfmacc_vv_f32m1(vImC, vImA, vReB, vl);
    __riscv_vse32_v_f32m1(&ReC[k], vReC, vl);
    __riscv_vse32_v_f32m1(&ImC[k], vImC, vl);
    k += (int)vl;
  }
}

/* SoA -> 镜像半复数: C[k]=ReC[k] (0..HALF); C[N-k]=ImC[k] (1..HALF-1) */
static void repack_mirrored(const float* ReC, const float* ImC, float* C) {
  int k = 0;
  while (k <= HALF) {
    size_t vl = __riscv_vsetvl_e32m1(HALF - k + 1);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(&ReC[k], vl);
    __riscv_vse32_v_f32m1(&C[k], v, vl);
    k += (int)vl;
  }
  k = 1;
  while (k < HALF) {
    size_t vl = __riscv_vsetvl_e32m1(HALF - k);
    vuint32m1_t idx = __riscv_vid_v_u32m1(vl);
    vuint32m1_t rev = __riscv_vrsub_vx_u32m1(idx, vl - 1, vl);
    vfloat32m1_t v = __riscv_vle32_v_f32m1(&ImC[k], vl);
    v = __riscv_vrgather_vv_f32m1(v, rev, vl);
    __riscv_vse32_v_f32m1(&C[N - k - (int)vl + 1], v, vl);
    k += (int)vl;
  }
}

static void fill_test_data(float* X) {
  /* 确定性伪随机频谱 (仅测试用, 与计时无关): 满足半复数布局约定 */
  X[0] = 1.5f; X[HALF] = -0.5f;
  for (int k = 1; k < HALF; k++) {
    X[k]     = (float)sin(k * 0.37) + 0.3f;   /* Re(F[k]) */
    X[N - k] = (float)cos(k * 0.21) - 0.1f;   /* Im(F[k]) */
  }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    float *A  = (float*)malloc(N * sizeof(float));
    float *B  = (float*)malloc(N * sizeof(float));
    float *C1 = (float*)malloc(N * sizeof(float));
    float *C2 = (float*)malloc(N * sizeof(float));
    float *ReA = (float*)malloc((HALF+1) * sizeof(float));
    float *ImA = (float*)malloc((HALF+1) * sizeof(float));
    float *ReB = (float*)malloc((HALF+1) * sizeof(float));
    float *ImB = (float*)malloc((HALF+1) * sizeof(float));
    float *ReC = (float*)malloc((HALF+1) * sizeof(float));
    float *ImC = (float*)malloc((HALF+1) * sizeof(float));
    if (!A || !B || !C1 || !C2 || !ReA || !ImA || !ReB || !ImB || !ReC || !ImC) {
        fprintf(stderr, "内存分配失败\n"); return 1;
    }

    fill_test_data(A);
    fill_test_data(B);

    /* 正确性: 两路径各跑一次并比对 */
    half_complex_mul_rvv(A, B, C1, N);
    unpack_soa(A, ReA, ImA);
    unpack_soa(B, ReB, ImB);
    mul_soa(ReA, ImA, ReB, ImB, ReC, ImC);
    repack_mirrored(ReC, ImC, C2);

    float max_diff = 0.0f;
    for (int k = 0; k < N; k++) {
        float d = fabsf(C1[k] - C2[k]);
        if (d > max_diff) max_diff = d;
    }
    printf("[check] 路径1(镜像) vs 路径2(SoA) 最大差: %.6e (期望 0 或末位舍入)\n",
           (double)max_diff);

    /* 计时: 路径 1 */
    struct timespec t0, t1;
    double secs;
    for (int r = 0; r < WARMUP; r++) half_complex_mul_rvv(A, B, C1, N);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) half_complex_mul_rvv(A, B, C1, N);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] 路径1 镜像布局直接乘 : reps=%d per_mul=%.3fus\n",
           REPS, secs / REPS * 1e6);

    /* 计时: 路径 2 (unpack x2 + 连续乘 + repack) */
    for (int r = 0; r < WARMUP; r++) {
        unpack_soa(A, ReA, ImA); unpack_soa(B, ReB, ImB);
        mul_soa(ReA, ImA, ReB, ImB, ReC, ImC);
        repack_mirrored(ReC, ImC, C2);
    }
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) {
        unpack_soa(A, ReA, ImA); unpack_soa(B, ReB, ImB);
        mul_soa(ReA, ImA, ReB, ImB, ReC, ImC);
        repack_mirrored(ReC, ImC, C2);
    }
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] 路径2 unpack+SoA乘+repack : reps=%d per_mul=%.3fus\n",
           REPS, secs / REPS * 1e6);

    /* 只计 SoA 连续乘本身 (对照: 理想情况下布局转换零成本时可达的时间) */
    for (int r = 0; r < WARMUP; r++) mul_soa(ReA, ImA, ReB, ImB, ReC, ImC);
    now_monotonic(&t0);
    for (int r = 0; r < REPS; r++) mul_soa(ReA, ImA, ReB, ImB, ReC, ImC);
    now_monotonic(&t1);
    secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("[time] (参照) SoA 连续乘本身  : reps=%d per_mul=%.3fus\n",
           REPS, secs / REPS * 1e6);

    free(A); free(B); free(C1); free(C2);
    free(ReA); free(ImA); free(ReB); free(ImB); free(ReC); free(ImC);
    return 0;
}
