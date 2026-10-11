/* ========================================================================
 * irdfft_main.c — 任务一 RVV 算子测试主程序 (往返测试 + Circulant 测试 + CLI) 
 *
 * 编译 (Linux 虚拟机, 与正变换文件一起链接): 
 *   riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static -O3 -o irdfft_rvv \
 *     irdfft_main.c irdfft_butterfly_compute.c irdfft_external.c \
 *     irdfft_half_complex_mul.c irdfft_circulant.c \
 *     rdfft_butterfly_compute.c external.c twiddle_factors.c -lm
 *
 * 运行:
 *   spike --isa=rv64gcv $HOME/riscv/riscv64-unknown-linux-gnu/bin/pk ./irdfft_rvv
 *
 * CLI 模式 (与 Python torch.fft 比对, 接口与标量版一致):
 *   ./irdfft_rvv <N> <w.bin> <x.bin> <y.bin>
 * ======================================================================== */
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N);
extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void execute_irdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void circulant_matmul_rvv_f32(const float* w, const float* x, float* y,
                                     const float* W_re, const float* W_im, int N);

/* 参考 DFT (直接按定义计算, O(N^2), 作为真值基准) */
static void reference_dft(const float* x, int N, float* ref_re, float* ref_im) {
    for (int k = 0; k <= N / 2; k++) {
        double sum_re = 0.0, sum_im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / (double)N;
            sum_re += x[n] * cos(angle);
            sum_im += x[n] * sin(angle);
        }
        ref_re[k] = (float)sum_re;
        ref_im[k] = (float)sum_im;
    }
}

/* 从半复数数组提取第 k 个频点的实部与虚部 */
static inline void get_half_complex(const float* Y, int N, int k,
                                    float* out_re, float* out_im) {
    if (k == 0) {
        *out_re = Y[0];
        *out_im = 0.0f;
    } else if (k == N / 2) {
        *out_re = Y[N / 2];
        *out_im = 0.0f;
    } else {
        *out_re = Y[k];
        *out_im = Y[N - k];
    }
}

/* 测试 rdFFT / irdFFT 往返精度 */
static void test_roundtrip(int N) {
    printf("\n===== RVV rdFFT/irdFFT 往返测试 (N=%d) =====\n", N);

    float* x      = (float*)malloc(N * sizeof(float));
    float* Y      = (float*)malloc(N * sizeof(float));
    float* W_re   = (float*)malloc((N / 2) * sizeof(float));
    float* W_im   = (float*)malloc((N / 2) * sizeof(float));
    float* ref_re = (float*)malloc((N / 2 + 1) * sizeof(float));
    float* ref_im = (float*)malloc((N / 2 + 1) * sizeof(float));

    /* 构造测试信号 */
    for (int n = 0; n < N; n++) {
        x[n] = 1.0f + 2.0f * (float)cos(2.0 * M_PI * n / (double)N)
               + 0.5f * (float)sin(4.0 * M_PI * n / (double)N);
        Y[n] = x[n];
    }

    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);

    /* 正向 rdFFT, 与参考 DFT 比对 */
    execute_rdfft_rvv_f32(Y, W_re, W_im, N);
    reference_dft(x, N, ref_re, ref_im);

    double max_err = 0.0, mse = 0.0;
    for (int k = 0; k <= N / 2; k++) {
        float re, im;
        get_half_complex(Y, N, k, &re, &im);
        double err_re = fabs((double)re - ref_re[k]);
        double err_im = fabs((double)im - ref_im[k]);
        if (err_re > max_err) max_err = err_re;
        if (err_im > max_err) max_err = err_im;
        mse += err_re * err_re + err_im * err_im;
    }
    mse /= (double)(N / 2 + 1);
    printf("  正向 rdFFT vs 参考 DFT: max_abs = %.6e, RMSE = %.6e\n",
           max_err, sqrt(mse));

    /* 逆向 irdFFT, 应恢复原始信号 (与正变换共用同一张 twiddle 表) */
    execute_irdfft_rvv_f32(Y, W_re, W_im, N);

    max_err = 0.0;
    mse = 0.0;
    for (int n = 0; n < N; n++) {
        double err = fabs((double)Y[n] - x[n]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= (double)N;
    printf("  逆向 irdFFT 恢复原始信号: max_abs = %.6e, RMSE = %.6e\n",
           max_err, sqrt(mse));

    free(x);
    free(Y);
    free(W_re);
    free(W_im);
    free(ref_re);
    free(ref_im);
}

/* 测试 Circulant 矩阵乘法并与直接循环卷积比对 */
static void test_circulant(int N) {
    printf("\n===== RVV Circulant 矩阵乘法测试 (N=%d) =====\n", N);

    float* w     = (float*)malloc(N * sizeof(float));
    float* x     = (float*)malloc(N * sizeof(float));
    float* y_fft = (float*)malloc(N * sizeof(float));
    float* y_ref = (float*)malloc(N * sizeof(float));
    float* W_re  = (float*)malloc((N / 2) * sizeof(float));
    float* W_im  = (float*)malloc((N / 2) * sizeof(float));

    /* 测试数据 (与标量版一致, 便于交叉比对) */
    for (int n = 0; n < N; n++) {
        w[n] = (float)((n + 1) % 5) - 2.0f;
        x[n] = (float)((n * 3) % 7) - 3.0f;
    }

    /* 直接循环卷积作为参考真值: y[n] = sum_k w[k] * x[(n-k) mod N] */
    for (int n = 0; n < N; n++) {
        double acc = 0.0;
        for (int k = 0; k < N; k++) {
            int idx = (n - k + N) % N;
            acc += (double)w[k] * x[idx];
        }
        y_ref[n] = (float)acc;
    }

    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);
    circulant_matmul_rvv_f32(w, x, y_fft, W_re, W_im, N);

    double max_err = 0.0, mse = 0.0;
    for (int n = 0; n < N; n++) {
        double err = fabs((double)y_fft[n] - y_ref[n]);
        if (err > max_err) max_err = err;
        mse += err * err;
    }
    mse /= (double)N;
    printf("  Circulant vs 直接循环卷积: max_abs = %.6e, RMSE = %.6e\n",
           max_err, sqrt(mse));

    free(w);
    free(x);
    free(y_fft);
    free(y_ref);
    free(W_re);
    free(W_im);
}

/* 命令行模式: 与 Python torch.fft 对比
 * 用法: ./irdfft_rvv <N> <w.bin> <x.bin> <y.bin>
 *   从 w.bin、x.bin 读取长度为 N 的 float32 数组,
 *   计算 Circulant 矩阵乘法 y = C(w)*x, 写入 y.bin。
 * 无参数时运行内置测试。 */
static int cli_mode(int argc, char** argv) {
    if (argc != 5) {
        return -1;
    }

    int N = atoi(argv[1]);
    if ((N & (N - 1)) != 0 || N < 2) {
        fprintf(stderr, "错误：N 必须是大于等于 2 的 2 的幂\n");
        return 1;
    }

    float* w    = (float*)malloc(N * sizeof(float));
    float* x    = (float*)malloc(N * sizeof(float));
    float* y    = (float*)malloc(N * sizeof(float));
    float* W_re = (float*)malloc((N / 2) * sizeof(float));
    float* W_im = (float*)malloc((N / 2) * sizeof(float));

    FILE* fw = fopen(argv[2], "rb");
    FILE* fx = fopen(argv[3], "rb");
    if (!fw || !fx) {
        fprintf(stderr, "错误：无法打开输入文件\n");
        return 1;
    }
    if (fread(w, sizeof(float), N, fw) != (size_t)N ||
        fread(x, sizeof(float), N, fx) != (size_t)N) {
        fprintf(stderr, "错误：输入文件长度不足\n");
        return 1;
    }
    fclose(fw);
    fclose(fx);

    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);
    circulant_matmul_rvv_f32(w, x, y, W_re, W_im, N);

    FILE* fy = fopen(argv[4], "wb");
    if (!fy) {
        fprintf(stderr, "错误：无法打开输出文件\n");
        return 1;
    }
    fwrite(y, sizeof(float), N, fy);
    fclose(fy);

    free(w);
    free(x);
    free(y);
    free(W_re);
    free(W_im);
    return 0;
}

int main(int argc, char** argv) {
    int rc = cli_mode(argc, argv);
    if (rc >= 0) {
        return rc;
    }

    /* 无参数时运行内置测试 */
    test_roundtrip(8);
    test_roundtrip(16);
    test_roundtrip(64);
    test_roundtrip(1024);

    test_circulant(8);
    test_circulant(16);
    test_circulant(64);
    test_circulant(256);

    printf("\n===== 内置测试结束 =====\n");
    printf("误差量级参考: float32 舍入 ~O(eps*log2(N)), max_abs 应在 1e-6~1e-5 量级\n");
    return 0;
}
