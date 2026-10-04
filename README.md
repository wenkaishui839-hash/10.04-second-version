# 大创项目归档：Block-Circulant NICE 模型端到端硬件部署

## 项目概述

本项目旨在实现基于分块循环矩阵（Block-Circulant Matrix）的 NICE 图像生成模型，并打通从 Python/PyTorch 顶层算法训练，到纯 C 语言底层推理，再到 RISC-V 向量扩展（RVV）硬件加速的**端到端（End-to-End）全栈部署流水线**。

目前项目第一阶段已圆满收官，团队成功将原版复杂度极高的全连接层替换为分块循环矩阵，并在 Spike 模拟器上完成了混合指令集的硬件推理闭环。

## 任务完成情况汇总

* **任务一：底层硬件加速算子开发（C/RVV）**
* 成功使用纯 C 语言及 RISC-V 向量扩展指令集（RVV）开发了快速傅里叶变换（FFT/iFFT）核心算子。
* 实现了底层内存的高效管理与张量频域乘法，为上层模型提供了高性能的底层数学加速库。


* **任务二 & 任务三：算法架构重构与模型训练（Python/PyTorch）**
* 构建了 `Block-Circulant NICE` 模型结构，成功将原有全连接层完全替换为分块循环矩阵层，大幅压缩了模型体积。
* 引入了自适应的 Zero-Padding（补零）机制，将 MNIST 图像数据维度精准对齐至 1024 维，完美契合底层基-2 FFT 的硬件规范。
* 编写了自动化导出脚本，可将训练后的循环矩阵权重及 Scaling（缩放层）系数无损导出为 C 语言可读的二进制 `.bin` 文件。


* **任务四：纯 C 推理引擎搭建与全链路整合（C 语言）**
* 脱离复杂的深度学习框架，使用纯 C 语言从零搭建了 NICE 模型的前向推理框架。
* 成功接入任务一的 RVV 加速算子，并编写了自动化权重解析脚本（`convert_weights.py` 与 `load_weights.c`）。
* 搭建了严格的自动化测试比对系统（Python vs C），验证了底层硬件输出张量与 Python 浮点运算结果的绝对一致性。



## 核心性能指标

本阶段在保证生成图像极其清晰的前提下，取得了以下突破性数据：

| 性能指标 | 测试结果 | 备注 |
| --- | --- | --- |
| **模型参数压缩率** | **99.50%** | 参数量由 padded 原版 20,995,072 暴降至 105,472（按 .bin 权重文件字节计 99.44%，按参数量计 99.50%；与 784 维原版 19,158,352 参数相比为 99.45%）。 |
| **端到端向量化率** | **15.06%** | 核心推理区间实测：RVV 版 19,615,671 指令中含 2,953,760 条向量指令；相比纯标量版 31,227,060 条减少 37.2%。 |
| **板上实测加速比** | **1.41×** | CanMV K230（C908 @1.6GHz，RVV 1.0）实测：RVV 13.5ms/图 vs 标量 19.0ms/图（关 WiFi 干净环境，RVV 三次平均）；算子级（单次 rdFFT(256)）为 1.12×（13.2µs vs 14.8µs）。 |
| **推理峰值内存** | **1.15 MB** | 板上与 Spike 双端 brk 实测一致（权重 426KB + 工作缓冲 + 堆栈开销）。 |
| **推理正确性** | **9.5e-07** | C 推理与 PyTorch 参考最大绝对误差，float32 舍入量级。 |
| **算法训练加速** | **提升 34%** | 同等 50 epoch 下实测：原版 NICE 671.6s → block-circulant NICE 443.0s（metrics.json 数据，已复核）。 |
| **维度对齐情况** | **1024维** | 完美适配底层硬件 FFT 对 $2^n$ 维度的数据要求。 |

## 系统运行流水线 (Pipeline)

本项目的运行逻辑已实现彻底解耦与无缝互通，分为三个独立阶段：

1. **上位机炼丹 (Training & Exporting)**
* 在配置 GPU 的计算机上运行 Python 统一实验脚本：
`python test_dachuang/run_nice_experiment.py --model-type padded-block-circ --epochs 50 --batch-size 1024 --data-dir data`
* 训练结束后自动导出循环矩阵权重与 Scaling 权重的 `.txt` / `.pt`，
  再用 `convert_weights.py` 将 `.txt` 转为 C 可读的二进制 `.bin`。
* 最终部署模型训练产物归档于 `assignment2 and 3/runs/padded_block_circ_nice_e50/`。

2. **仿真验证 (Spike 模拟器)**
* 用 `riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3` 编译 `nice_infer_rvv`
  （纯标量基线 `nice_infer_scalar` 用 `-march=rv64gc`）。
* 执行 `./bench.sh` 一键完成指令统计 / 正确性比对 / 内存测量。

3. **真机部署 (RISC-V 板子)**
* 将 `nice_infer_rvv` / `nice_infer_scalar` 与两个 `.bin`、`z_input.txt` 拷贝至
  CanMV K230（C908，RVV 1.0）板载 Linux。
* 执行 `taskset -c 0 ./nice_infer_rvv circulant_weights.bin scaling_weights.bin z_input.txt 100`
  输出图像与 `[time]`（单图耗时）/ `[heap]`（峰值堆）实测数据。
