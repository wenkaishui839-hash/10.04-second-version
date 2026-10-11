#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern void execute_rdfft_rvv_f32(float* Y, const float* W_re, const float* W_im, int N);
extern void generate_rdfft_twiddle_factors_staged(float* W_re, float* W_im, int N);

/**
 * 参考 DFT (直接按定义计算, O(N²), 作为真值基准)
 * 结果按半复数格式存入 ref_re[k], ref_im[k], k = 0 .. N/2
 */
void reference_dft(const float* x, int N, float* ref_re, float* ref_im) {
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

int main() {
    // ★ 改这个 N 来测试不同规模 (必须是 2 的幂次方)
    const int N = 8;

    float* x       = (float*)malloc(N * sizeof(float));
    float* Y       = (float*)malloc(N * sizeof(float));
    float* W_re    = (float*)malloc((N / 2) * sizeof(float));
    float* W_im    = (float*)malloc((N / 2) * sizeof(float));
    float* ref_re  = (float*)malloc((N / 2 + 1) * sizeof(float));
    float* ref_im  = (float*)malloc((N / 2 + 1) * sizeof(float));

    // 构造输入: x[n] = 1.0 + 2.0*cos(2*pi*n/N)
    printf("=== rdFFT RVV Error Analysis (N=%d) ===\n\n", N);
    printf("Input: x[n] = 1.0 + 2.0*cos(2*pi*n/%d)\n\n", N);
    for (int n = 0; n < N; n++) {
        x[n] = 1.0f + 2.0f * (float)cos(2.0 * M_PI * n / (double)N);
        Y[n] = x[n];
    }

    // 1. 计算参考 DFT (真值)
    reference_dft(x, N, ref_re, ref_im);

    // 2. rdFFT 旋转因子 + 执行
    generate_rdfft_twiddle_factors_staged(W_re, W_im, N);
    execute_rdfft_rvv_f32(Y, W_re, W_im, N);

    // 3. 逐频点对比
    double max_abs_err = 0.0;
    double mse = 0.0;
    int count = 0;

    printf("%6s  %12s  %12s  %12s  %12s  %12s  %12s\n",
           "k", "Ref_Re", "rdFFT_Re", "Err_Re",
           "Ref_Im", "rdFFT_Im", "Err_Im");
    printf("------  ------------  ------------  ------------  ------------  ------------  ------------\n");

    for (int k = 0; k <= N / 2; k++) {
        float rdfft_re, rdfft_im;

        if (k == 0) {
            rdfft_re = Y[0];  rdfft_im = 0.0f;
        } else if (k == N / 2) {
            rdfft_re = Y[N / 2];  rdfft_im = 0.0f;
        } else {
            rdfft_re = Y[k];  rdfft_im = Y[N - k];
        }

        float err_re = (float)fabs((double)rdfft_re - ref_re[k]);
        float err_im = (float)fabs((double)rdfft_im - ref_im[k]);

        const char* label = (k == 0) ? "DC" : (k == N / 2) ? "Nyq" : "";
        printf("%3d%3s %14.6f  %14.6f  %14.6e  %14.6f  %14.6f  %14.6e\n",
               k, label,
               (double)ref_re[k], (double)rdfft_re, (double)err_re,
               (double)ref_im[k], (double)rdfft_im, (double)err_im);

        double err = err_re * err_re + err_im * err_im;
        if (err_re > max_abs_err) max_abs_err = err_re;
        if (err_im > max_abs_err) max_abs_err = err_im;
        mse += err;
        count++;
    }

    mse /= count;

    printf("\n=== Error Summary ===\n");
    printf("  Frequency bins compared: %d\n", count);
    printf("  Max absolute error:      %.6e\n", max_abs_err);
    printf("  MSE:                     %.6e\n", mse);
    printf("  RMSE:                    %.6e\n", sqrt(mse));
    printf("  (float32 roundoff, expected ~ O(eps * log2(N)) = ~%.1e)\n",
           1.0e-7 * log2(N));

    free(x); free(Y); free(W_re); free(W_im); free(ref_re); free(ref_im);
    return 0;
}
