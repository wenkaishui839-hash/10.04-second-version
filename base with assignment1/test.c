#include <stdio.h>

// 假设我们现在处理的序列长度是 8 (N=8)
#define N 8

int main() {
    // 假设这是我们的内存空间，长度正好是 8，没有多余的缓存！
    // 里面的数字代表运行过程中的数据
    float memory[N] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0};

    printf("--- rdFFT (基于论文) 内存存取位置揭秘 ---\n\n");

    // 1. 论文提到的特殊点：0 和 N/2 位置是纯实数
    printf("[特殊点] 频率 0 的实部存在 memory[0]   = %.1f\n", memory[0]);
    printf("[特殊点] 频率 %d 的实部存在 memory[%d]   = %.1f\n", N/2, N/2, memory[N/2]);
    printf("----------------------------------------\n");

    // 2. 论文的核心创新：复数对的“首尾交替”存储法
    for (int k = 1; k < N/2; k++) {
        printf("当计算频率点 k = %d 时：\n", k);
        // 实部正着存
        printf("  -> 去找实部：索引 k = %d,   读取 memory[%d] = %.1f\n", k, k, memory[k]);
        // 虚部倒着存
        printf("  -> 去找虚部：索引 N-k = %d, 读取 memory[%d] = %.1f\n\n", N-k, N-k, memory[N-k]);
    }

    return 0;
}