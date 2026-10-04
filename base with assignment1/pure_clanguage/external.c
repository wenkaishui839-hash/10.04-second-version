#include <stdint.h>

// 声明外部的标量标量蝶形计算函数
extern void rdfft_butterfly_compute_scalar(float* Y, const float* W_re, const float* W_im, int N, int half_step, int base);

static void bit_reverse_permutation(float* Y, int N) {
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            float temp = Y[i];
            Y[i] = Y[j];
            Y[j] = temp;
        }
        int k = N / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

void execute_rdfft_scalar(float* Y, const float* W_re, const float* W_im, int N) {
    bit_reverse_permutation(Y, N);
    for (int half_step = 1; half_step < N; half_step *= 2) {
        int step = 2 * half_step;
        for (int base = 0; base < N; base += step) {
            // 这里调用我们即将编写的纯 C 标量版本内核
            rdfft_butterfly_compute_scalar(Y, W_re, W_im, N, half_step, base);
        }
    }
}