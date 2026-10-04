#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * 公共接口声明
 * ======================================================================== */

/* 实数 FFT 参数结构：半复数格式长度为 N，旋转因子表长度为 N/2 */
#define MAX_FFT_SIZE (1 << 20)

/* 位反转置换：将长度为 N（N 为 2 的幂）的实数数组按位反转重排 */
void bit_reverse_permute_float(float *arr, int N);

/* 生成正向 rdFFT 旋转因子：W_N^k = exp(-j*2*pi*k/N), k = 0..N/2-1 */
void generate_rdfft_twiddle_factors(float *W_re, float *W_im, int N);

/* 生成逆向 irdFFT 旋转因子：W_N^{-k} = exp(+j*2*pi*k/N), k = 0..N/2-1 */
void generate_irdfft_twiddle_factors(float *W_re, float *W_im, int N);

/* 正向实数 FFT：输入输出均为半复数格式，原地计算 */
void execute_rdfft_f32(float *Y, const float *W_re, const float *W_im, int N);

/* 逆向实数 FFT：输入输出均为半复数格式，原地计算，结果已除以 N */
void execute_irdfft_f32(float *Y, const float *W_re, const float *W_im, int N);

/* 半复数格式逐元素复数乘法：Z = X .* Y */
void half_complex_mul(const float *X, const float *Y, float *Z, int N);

/* Circulant 矩阵乘法：y = C(w) * x，基于 FFT 实现 */
void circulant_mul_f32(const float *w, const float *x, float *y,
                       const float *W_re, const float *W_im,
                       const float *IW_re, const float *IW_im,
                       int N);

/* 辅助：从半复数数组中提取第 k 个频点的实部与虚部 */
static inline void get_half_complex(const float *Y, int N, int k,
                                    float *out_re, float *out_im) {
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

/* 辅助：将第 k 个频点的实部与虚部写回半复数数组 */
static inline void set_half_complex(float *Y, int N, int k,
                                    float re, float im) {
  if (k == 0) {
    Y[0] = re;
  } else if (k == N / 2) {
    Y[N / 2] = re;
  } else {
    Y[k] = re;
    Y[N - k] = im;
  }
}

/* ========================================================================
 * 工具函数：位反转与旋转因子表生成
 * ======================================================================== */

/* 位反转置换：按二进制位反转重新排列数组元素
 * 例如 N=8 时，索引 1(001b) 反转后为 4(100b) */
void bit_reverse_permute_float(float *arr, int N) {
  int bits = 0;
  int tmp = N;
  while (tmp > 1) {
    bits++;
    tmp >>= 1;
  }

  for (int i = 0; i < N; i++) {
    int j = 0;
    for (int b = 0; b < bits; b++) {
      j = (j << 1) | ((i >> b) & 1);
    }
    if (j > i) {
      float t = arr[i];
      arr[i] = arr[j];
      arr[j] = t;
    }
  }
}

/* 正向 rdFFT 旋转因子：angle = -2*pi*k/N */
void generate_rdfft_twiddle_factors(float *W_re, float *W_im, int N) {
  int max_k = N / 2;
  for (int k = 0; k < max_k; k++) {
    double angle = -2.0 * M_PI * (double)k / (double)N;
    W_re[k] = (float)cos(angle);
    W_im[k] = (float)sin(angle);
  }
}

/* 逆向 irdFFT 旋转因子：angle = +2*pi*k/N（正向因子的共轭） */
void generate_irdfft_twiddle_factors(float *W_re, float *W_im, int N) {
  int max_k = N / 2;
  for (int k = 0; k < max_k; k++) {
    double angle = 2.0 * M_PI * (double)k / (double)N;
    W_re[k] = (float)cos(angle);
    W_im[k] = (float)sin(angle);
  }
}

/* ========================================================================
 * 正向 rdFFT 蝶形与执行函数
 * ======================================================================== */

/* 正向 rdFFT 单级共轭对蝶形运算
 *
 * 将两个相邻的半复数子块 E（大小 M）和 O（大小 M）合并为大小 2M 的半复数子块。
 * 半复数映射规则：
 *   Re(X[k])      -> Y[base + k]
 *   Im(X[k])      -> Y[base + 2*M - k]   (k = 1..M-1)
 *   Re(X[0])      -> Y[base]
 *   Re(X[M])      -> Y[base + M]         (Nyquist)
 */
static void rdfft_butterfly_compute(float *Y, const float *W_re,
                                    const float *W_im, int N, int M,
                                    int base) {
  int twiddle_stride = N / (2 * M);

  /* Case 1: k = 0，直流分量，纯实数加减 */
  {
    float E_re = Y[base];
    float O_re = Y[base + M];
    Y[base] = E_re + O_re;
    Y[base + M] = E_re - O_re;
  }

  /* Case 2: k = M/2，子块奈奎斯特频率，仅当 M >= 2 时存在
   * X[M/2] = E[M/2] - j*O[M/2]
   * 实部写入 Y[base + M/2]，虚部写入 Y[base + 2*M - M/2] = Y[base + 3*M/2] */
  if (M >= 2) {
    int k = M / 2;
    float E_re = Y[base + k];
    float O_re = Y[base + M + k];
    Y[base + k] = E_re;
    Y[base + 2 * M - k] = -O_re;
  }

  /* Case 3: 一般频点 k = 1..M/2-1，共轭对蝶形 */
  for (int k = 1; k < M / 2; k++) {
    /* 按半复数格式提取 E[k] 与 O[k] 的实部、虚部 */
    float e_re = Y[base + k];
    float e_im = Y[base + M - k];
    float o_re = Y[base + M + k];
    float o_im = Y[base + 2 * M - k];

    /* 旋转因子 W_{2M}^k = W_N^{k * stride} */
    int tw_idx = k * twiddle_stride;
    float w_re = W_re[tw_idx];
    float w_im = W_im[tw_idx];

    /* T = W * O */
    float t_re = w_re * o_re - w_im * o_im;
    float t_im = w_re * o_im + w_im * o_re;

    /* X[k] = E + T */
    float out_xk_re = e_re + t_re;
    float out_xk_im = e_im + t_im;

    /* X[M-k] = (E - T)*，即实部为 e_re - t_re，虚部为 t_im - e_im */
    float out_xmk_re = e_re - t_re;
    float out_xmk_im = t_im - e_im;

    /* 写回半复数格式 */
    Y[base + k] = out_xk_re;
    Y[base + M + k] = out_xmk_im;
    Y[base + M - k] = out_xmk_re;
    Y[base + 2 * M - k] = out_xk_im;
  }
}

/* 执行完整正向 rdFFT
 * 输入：Y 已初始化为时域实数序列 x[0..N-1]
 * 输出：Y 中存放半复数格式的频谱 X[0..N/2]
 */
void execute_rdfft_f32(float *Y, const float *W_re, const float *W_im, int N) {
  /* Cooley-Tukey DIT：先位反转，再逐级蝶形 */
  bit_reverse_permute_float(Y, N);

  for (int M = 1; M < N; M <<= 1) {
    for (int base = 0; base < N; base += (2 * M)) {
      rdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
    }
  }
}

/* ========================================================================
 * 逆向 irdFFT 蝶形与执行函数
 * ======================================================================== */

/* 逆向 rdFFT 单级共轭对蝶形运算
 *
 * 与正向蝶形互逆：将大小 2M 的半复数子块拆分为两个相邻的大小 M 子块。
 * 核心公式（含 1/2 因子的反向，即结果放大 2 倍，最终统一除以 N）：
 *   2*E[k] = X[k] + X[M-k]*
 *   2*O[k] = (X[k] - X[M-k]*) * W_{2M}^{-k}
 *
 * 其中 W_{2M}^{-k} = exp(+j*pi*k/M) 为正向旋转因子的共轭。
 */
static void irdfft_butterfly_compute(float *Y, const float *W_re,
                                     const float *W_im, int N, int M,
                                     int base) {
  int twiddle_stride = N / (2 * M);

  /* Case 1: k = 0，直流分量
   * 2*E[0] = X[0] + X[M]
   * 2*O[0] = X[0] - X[M] */
  {
    float X0 = Y[base];
    float XM = Y[base + M];
    Y[base] = X0 + XM;
    Y[base + M] = X0 - XM;
  }

  /* Case 2: k = M/2，子块奈奎斯特频率
   * 2*E[M/2] = 2*Re(X[M/2])
   * 2*O[M/2] = -2*Im(X[M/2]) */
  if (M >= 2) {
    int k = M / 2;
    float X_re = Y[base + k];
    float X_im = Y[base + 2 * M - k]; /* 这是 X[M/2] 的虚部存放位置 */
    Y[base + k] = 2.0f * X_re;
    Y[base + 2 * M - k] = -2.0f * X_im;
  }

  /* Case 3: 一般频点 k = 1..M/2-1 */
  for (int k = 1; k < M / 2; k++) {
    /* 按半复数格式提取 X[k] 与 X[M-k] */
    float a = Y[base + k];          /* Re(X[k]) */
    float b = Y[base + 2 * M - k];  /* Im(X[k]) */
    float c = Y[base + M - k];      /* Re(X[M-k]) */
    float d = Y[base + M + k];      /* Im(X[M-k]) */

    /* 逆向旋转因子 W_{2M}^{-k} = W_N^{-k*stride}（共轭因子） */
    int tw_idx = k * twiddle_stride;
    float w_re = W_re[tw_idx];
    float w_im = W_im[tw_idx];

    /* 2*E[k] = X[k] + X[M-k]* = (a+c) + j(b-d) */
    float out_ek_re = a + c;
    float out_ek_im = b - d;

    /* 2*O[k] = (X[k] - X[M-k]*) * W_{2M}^{-k}
     * X[k] - X[M-k]* = (a-c) + j(b+d)
     * 乘以 (w_re + j*w_im) */
    float diff_re = a - c;
    float diff_im = b + d;
    float out_ok_re = diff_re * w_re - diff_im * w_im;
    float out_ok_im = diff_re * w_im + diff_im * w_re;

    /* 写回 E 块与 O 块的半复数格式 */
    Y[base + k] = out_ek_re;
    Y[base + M - k] = out_ek_im;
    Y[base + M + k] = out_ok_re;
    Y[base + 2 * M - k] = out_ok_im;
  }
}

/* 执行完整逆向 irdFFT
 * 输入：Y 为半复数格式的频谱 X[0..N/2]
 * 输出：Y 中存放时域实数序列 x[0..N-1]，已除以 N 归一化
 */
void execute_irdfft_f32(float *Y, const float *W_re, const float *W_im, int N) {
  /* 逆向 FFT 为 DIF 结构：先逐级蝶形，再位反转 */
  for (int M = N / 2; M >= 1; M >>= 1) {
    for (int base = 0; base < N; base += (2 * M)) {
      irdfft_butterfly_compute(Y, W_re, W_im, N, M, base);
    }
  }

  /* 位反转得到正常顺序 */
  bit_reverse_permute_float(Y, N);

  /* 每级蝶形均放大 2 倍，log2(N) 级共放大 N 倍，故除以 N 归一化 */
  float inv_N = 1.0f / (float)N;
  for (int i = 0; i < N; i++) {
    Y[i] *= inv_N;
  }
}

/* ========================================================================
 * 半复数逐元素乘法
 * ======================================================================== */

/* 半复数格式逐元素复数乘法：Z = X .* Y
 *
 * 半复数映射：
 *   Z[0]      = Re(Z[0])      （直流，纯实数）
 *   Z[N/2]    = Re(Z[N/2])    （奈奎斯特，纯实数）
 *   Z[k]      = Re(Z[k])      （k = 1..N/2-1）
 *   Z[N-k]    = Im(Z[k])      （倒序虚部）
 */
void half_complex_mul(const float *X, const float *Y, float *Z, int N) {
  /* 直流分量：两个实数相乘 */
  Z[0] = X[0] * Y[0];

  /* 奈奎斯特分量：两个实数相乘 */
  Z[N / 2] = X[N / 2] * Y[N / 2];

  /* 普通频点：逐对复数相乘 */
  for (int k = 1; k < N / 2; k++) {
    float a = X[k];
    float b = X[N - k];
    float c = Y[k];
    float d = Y[N - k];

    float re = a * c - b * d;
    float im = a * d + b * c;

    Z[k] = re;
    Z[N - k] = im;
  }
}

/* ========================================================================
 * Circulant 矩阵乘法
 * ======================================================================== */

/* Circulant 矩阵乘法算子：y = C(w) * x
 *
 * C(w) 是第一行为 w[0..N-1] 的循环矩阵：
 *   [ w0 w1 w2 ... w_{N-1} ]
 *   [ w_{N-1} w0 w1 ... w_{N-2} ]
 *   ...
 * 矩阵向量乘等价于 w 与 x 的循环卷积，可用 FFT 加速：
 *   Y = irdFFT(rdFFT(W) .* rdFFT(X))
 */
void circulant_mul_f32(const float *w, const float *x, float *y,
                       const float *W_re, const float *W_im,
                       const float *IW_re, const float *IW_im,
                       int N) {
  /* 分配两个临时半复数缓冲区 */
  float *buf_w = (float *)malloc(N * sizeof(float));
  float *buf_x = (float *)malloc(N * sizeof(float));

  /* 复制输入到时域缓冲区 */
  memcpy(buf_w, w, N * sizeof(float));
  memcpy(buf_x, x, N * sizeof(float));

  /* 频域变换 */
  execute_rdfft_f32(buf_w, W_re, W_im, N);
  execute_rdfft_f32(buf_x, W_re, W_im, N);

  /* 频域逐元素相乘 */
  half_complex_mul(buf_w, buf_x, buf_w, N);

  /* 逆变换回时域 */
  execute_irdfft_f32(buf_w, IW_re, IW_im, N);

  /* 复制结果到输出 */
  memcpy(y, buf_w, N * sizeof(float));

  free(buf_w);
  free(buf_x);
}

/* ========================================================================
 * 测试框架与 main
 * ======================================================================== */

/* 参考 DFT：仅计算 k = 0..N/2 的实部与虚部，作为误差基准 */
static void reference_dft(const float *x, int N, float *ref_re, float *ref_im) {
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

/* 测试 rdFFT / irdFFT 往返精度 */
static void test_roundtrip(int N) {
  printf("\n===== FFT 往返测试 (N=%d) =====\n", N);

  float *x = (float *)malloc(N * sizeof(float));
  float *Y = (float *)malloc(N * sizeof(float));
  float *W_re = (float *)malloc((N / 2) * sizeof(float));
  float *W_im = (float *)malloc((N / 2) * sizeof(float));
  float *IW_re = (float *)malloc((N / 2) * sizeof(float));
  float *IW_im = (float *)malloc((N / 2) * sizeof(float));
  float *ref_re = (float *)malloc((N / 2 + 1) * sizeof(float));
  float *ref_im = (float *)malloc((N / 2 + 1) * sizeof(float));

  /* 构造测试信号 */
  for (int n = 0; n < N; n++) {
    x[n] = 1.0f + 2.0f * (float)cos(2.0 * M_PI * n / (double)N)
           + 0.5f * (float)sin(4.0 * M_PI * n / (double)N);
    Y[n] = x[n];
  }

  generate_rdfft_twiddle_factors(W_re, W_im, N);
  generate_irdfft_twiddle_factors(IW_re, IW_im, N);

  /* 正向 rdFFT */
  execute_rdfft_f32(Y, W_re, W_im, N);

  /* 与参考 DFT 比对 */
  reference_dft(x, N, ref_re, ref_im);
  printf("\n正向 rdFFT 误差：\n");
  printf("%6s %14s %14s %14s %14s\n", "k", "Ref_Re", "FFT_Re", "Ref_Im",
         "FFT_Im");
  double max_err = 0.0, mse = 0.0;
  for (int k = 0; k <= N / 2; k++) {
    float re, im;
    get_half_complex(Y, N, k, &re, &im);
    double err_re = fabs((double)re - ref_re[k]);
    double err_im = fabs((double)im - ref_im[k]);
    if (err_re > max_err) max_err = err_re;
    if (err_im > max_err) max_err = err_im;
    mse += err_re * err_re + err_im * err_im;
    printf("%6d %14.6f %14.6f %14.6f %14.6f\n", k, ref_re[k], re, ref_im[k],
           im);
  }
  mse /= (double)(N / 2 + 1);
  printf("Max abs err = %.6e, RMSE = %.6e\n", max_err, sqrt(mse));

  /* 逆向 irdFFT，应恢复原始信号 */
  execute_irdfft_f32(Y, IW_re, IW_im, N);
  printf("\n逆向 irdFFT 恢复误差：\n");
  max_err = 0.0;
  mse = 0.0;
  for (int n = 0; n < N; n++) {
    double err = fabs((double)Y[n] - x[n]);
    if (err > max_err) max_err = err;
    mse += err * err;
  }
  mse /= (double)N;
  printf("Max abs err = %.6e, RMSE = %.6e\n", max_err, sqrt(mse));

  free(x);
  free(Y);
  free(W_re);
  free(W_im);
  free(IW_re);
  free(IW_im);
  free(ref_re);
  free(ref_im);
}

/* 测试 Circulant 矩阵乘法并与直接循环卷积比对 */
static void test_circulant(int N) {
  printf("\n===== Circulant 矩阵乘法测试 (N=%d) =====\n", N);

  float *w = (float *)malloc(N * sizeof(float));
  float *x = (float *)malloc(N * sizeof(float));
  float *y_fft = (float *)malloc(N * sizeof(float));
  float *y_ref = (float *)malloc(N * sizeof(float));
  float *W_re = (float *)malloc((N / 2) * sizeof(float));
  float *W_im = (float *)malloc((N / 2) * sizeof(float));
  float *IW_re = (float *)malloc((N / 2) * sizeof(float));
  float *IW_im = (float *)malloc((N / 2) * sizeof(float));

  /* 随机填充测试数据 */
  for (int n = 0; n < N; n++) {
    w[n] = (float)((n + 1) % 5) - 2.0f;
    x[n] = (float)((n * 3) % 7) - 3.0f;
  }

  /* 直接循环卷积作为参考真值 */
  for (int n = 0; n < N; n++) {
    double acc = 0.0;
    for (int k = 0; k < N; k++) {
      int idx = (n - k + N) % N;
      acc += (double)w[k] * x[idx];
    }
    y_ref[n] = (float)acc;
  }

  generate_rdfft_twiddle_factors(W_re, W_im, N);
  generate_irdfft_twiddle_factors(IW_re, IW_im, N);

  /* FFT 方法计算 Circulant 矩阵乘法 */
  circulant_mul_f32(w, x, y_fft, W_re, W_im, IW_re, IW_im, N);

  printf("%6s %14s %14s %14s\n", "n", "Ref", "FFT", "Err");
  double max_err = 0.0, mse = 0.0;
  for (int n = 0; n < N; n++) {
    double err = fabs((double)y_fft[n] - y_ref[n]);
    if (err > max_err) max_err = err;
    mse += err * err;
    printf("%6d %14.6f %14.6f %14.6e\n", n, y_ref[n], y_fft[n], err);
  }
  mse /= (double)N;
  printf("Circulant Max abs err = %.6e, RMSE = %.6e\n", max_err, sqrt(mse));

  free(w);
  free(x);
  free(y_fft);
  free(y_ref);
  free(W_re);
  free(W_im);
  free(IW_re);
  free(IW_im);
}

/* 命令行模式：用于与 Python torch.fft 对比
 * 用法: ./fft_test <N> <w.bin> <x.bin> <y.bin>
 *   从 w.bin、x.bin 读取长度为 N 的 float32 数组，
 *   计算 Circulant 矩阵乘法 y = C(w)*x，写入 y.bin。
 * 无参数时运行内置测试。
 */
static int cli_mode(int argc, char **argv) {
  if (argc != 5) {
    return -1;
  }

  int N = atoi(argv[1]);
  if ((N & (N - 1)) != 0 || N < 2) {
    fprintf(stderr, "错误：N 必须是大于等于 2 的 2 的幂\n");
    return 1;
  }

  float *w = (float *)malloc(N * sizeof(float));
  float *x = (float *)malloc(N * sizeof(float));
  float *y = (float *)malloc(N * sizeof(float));
  float *W_re = (float *)malloc((N / 2) * sizeof(float));
  float *W_im = (float *)malloc((N / 2) * sizeof(float));
  float *IW_re = (float *)malloc((N / 2) * sizeof(float));
  float *IW_im = (float *)malloc((N / 2) * sizeof(float));

  FILE *fw = fopen(argv[2], "rb");
  FILE *fx = fopen(argv[3], "rb");
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

  generate_rdfft_twiddle_factors(W_re, W_im, N);
  generate_irdfft_twiddle_factors(IW_re, IW_im, N);
  circulant_mul_f32(w, x, y, W_re, W_im, IW_re, IW_im, N);

  FILE *fy = fopen(argv[4], "wb");
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
  free(IW_re);
  free(IW_im);
  return 0;
}

int main(int argc, char **argv) {
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

  return 0;
}
