#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern void execute_rdfft_scalar(float* Y, const float* W_re, const float* W_im, int N);
extern void generate_rdfft_twiddle_factors(float* W_re, float* W_im, int N);

/**
 * 参考 DFT (直接按定义计算, O(N²))
 * 结果存入 ref_re[k], ref_im[k] — 半复数格式: k = 0 .. N/2
 */
void reference_dft(const float* x, int N, float* ref_re, float* ref_im) {
    for (int k = 0; k <= N / 2; k++) {
        double sum_re = 0.0, sum_im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            sum_re += x[n] * cos(angle);
            sum_im += x[n] * sin(angle);
        }
        ref_re[k] = (float)sum_re;
        ref_im[k] = (float)sum_im;
    }
}

int main() {
    // 可修改 N 来测试不同规模
    const int N = 8;

    float* x       = (float*)malloc(N * sizeof(float));
    float* Y       = (float*)malloc(N * sizeof(float));
    float* W_re    = (float*)malloc((N / 2) * sizeof(float));
    float* W_im    = (float*)malloc((N / 2) * sizeof(float));
    float* ref_re  = (float*)malloc((N / 2 + 1) * sizeof(float));
    float* ref_im  = (float*)malloc((N / 2 + 1) * sizeof(float));

    // 构造输入信号: x[n] = 1.0 + 2.0*cos(2*pi*n/N)
    printf("=== rdFFT Error Analysis (N=%d) ===\n\n", N);
    printf("Input: x[n] = 1.0 + 2.0*cos(2*pi*n/%d)\n\n", N);
    for (int n = 0; n < N; n++) {
        x[n] = 1.0f + 2.0f * (float)cos(2.0 * M_PI * n / N);
        Y[n] = x[n];  // 复制到 Y 供 rdFFT 原地计算
    }

    // 1. 计算参考 DFT
    reference_dft(x, N, ref_re, ref_im);

    // 2. 生成旋转因子并执行 rdFFT
    generate_rdfft_twiddle_factors(W_re, W_im, N);
    execute_rdfft_scalar(Y, W_re, W_im, N);

    // 3. 逐频点对比、统计误差
    double max_err_re = 0.0, max_err_im = 0.0;
    double mse_re = 0.0, mse_im = 0.0;
    int count = 0;

    printf("%6s  %12s  %12s  %12s  %12s  %12s\n",
           "k", "Ref_Re", "rdFFT_Re", "Err_Re",
           "Ref_Im", "rdFFT_Im", "Err_Im");
    printf("------  ------------  ------------  ------------  ------------  ------------  ------------\n");

    // k = 0 (DC)
    {
        float err_re = (float)fabs(Y[0] - ref_re[0]);
        printf("DC  %14.6f  %14.6f  %14.6e  %14.6f  %14.6f  %14.6e\n",
               (double)ref_re[0], (double)Y[0], (double)err_re,
               (double)ref_im[0], 0.0, (double)fabs(ref_im[0]));
        if (err_re > max_err_re) max_err_re = err_re;
        mse_re += err_re * err_re;
        count++;
    }

    // k = 1 .. N/2-1
    for (int k = 1; k < N / 2; k++) {
        float rdfft_re = Y[k];
        float rdfft_im = Y[N - k];
        float err_re = (float)fabs(rdfft_re - ref_re[k]);
        float err_im = (float)fabs(rdfft_im - ref_im[k]);

        printf("%3d  %14.6f  %14.6f  %14.6e  %14.6f  %14.6f  %14.6e\n",
               k, (double)ref_re[k], (double)rdfft_re, (double)err_re,
               (double)ref_im[k], (double)rdfft_im, (double)err_im);

        if (err_re > max_err_re) max_err_re = err_re;
        if (err_im > max_err_im) max_err_im = err_im;
        mse_re += err_re * err_re;
        mse_im += err_im * err_im;
        count++;
    }

    // k = N/2 (Nyquist)
    {
        float err_re = (float)fabs(Y[N / 2] - ref_re[N / 2]);
        printf("Nyq %14.6f  %14.6f  %14.6e  %14.6f  %14.6f  %14.6e\n",
               (double)ref_re[N / 2], (double)Y[N / 2], (double)err_re,
               (double)ref_im[N / 2], 0.0, (double)fabs(ref_im[N / 2]));
        if (err_re > max_err_re) max_err_re = err_re;
        mse_re += err_re * err_re;
        count++;
    }

    mse_re /= count;
    mse_im /= count;

    printf("\n=== Error Summary ===\n");
    printf("  Frequency bins compared: %d\n", count);
    printf("  Max absolute error (Re):  %.6e\n", max_err_re);
    printf("  Max absolute error (Im):  %.6e\n", max_err_im);
    printf("  MSE (Re):                 %.6e\n", mse_re);
    printf("  MSE (Im):                 %.6e\n", mse_im);
    printf("  RMSE (Re):                %.6e\n", sqrt(mse_re));
    printf("  RMSE (Im):                %.6e\n", sqrt(mse_im));
    printf("  (Errors are due to IEEE 754 float32 roundoff accumulation)\n");

    free(x); free(Y); free(W_re); free(W_im); free(ref_re); free(ref_im);
    return 0;
}
