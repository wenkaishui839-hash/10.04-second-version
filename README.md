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
| **端到端向量化率** | **25.61%** | 核心推理区间实测：RVV 版 5,577,751 指令中含 1,428,640 条向量指令；相比纯标量版 21,742,420 条减少 74.3%（v4 codelet 特化后；v3 为 11,823,831 / 2,057,760 / 17.40%，v2 为 16,951,031 / 45.7%）。 |
| **板上实测加速比** | **2.63×** | CanMV K230（C908 @1.6GHz，RVV 1.0）实测：RVV 5.14ms/图 vs 标量 13.51ms/图（v4 codelet 特化后；v3 为 1.70×，v2 为 1.74×，优化前为 1.41×）；算子级（单次 rdFFT(256)）**2.34×**（6.37 vs 14.89µs），并同板反超纯标量 kissfft 80%。 |
| **推理峰值内存** | **1.39 MB** | 板上与 Spike 双端 brk 实测一致（权重 426KB + 权重频谱缓存 264KB + 常驻工作区；v4 起前向过程零 malloc，brk before=after 为实测证据；比 FFTW 版 1.84MB 省 24%）。 |
| **推理正确性** | **1.2e-06** | C 推理与 PyTorch 参考最大绝对误差（RVV 1.19e-06 / 标量 7.75e-07），float32 舍入量级；v4 codelet 改变了浮点结合顺序，舍入路径与 v2/v3 略有不同但量级不变。 |
| **算法训练加速** | **提升 34%** | 同等 50 epoch 下实测：原版 NICE 671.6s → block-circulant NICE 443.0s（metrics.json 数据，已复核）。 |
| **维度对齐情况** | **1024维** | 完美适配底层硬件 FFT 对 $2^n$ 维度的数据要求。 |

## 与现有 FFT 库的对比结论（诚实性说明）

同板（K230）实测，三层口径完整：
- **torch.fft**（x86 精度对齐）：误差 1e-6 量级，全部 PASS（任务一表 1）；
- **kissfft**（同板性能）：本组 RVV rdFFT v4 6.37µs，**反超纯标量 kissfft 80.1%**（v2 时为 6.4%）；
- **FFTW**（同板性能，算子级）：本组 RVV v4 6.37µs，**仍慢于 FFTW 标量（3.87µs）1.65×**
  （v2 为 2.7×，codelet 特化追回一半差距）——根因是 FFTW 预生成 codelet 的
  代码质量（其标量比本组标量快 3.85×）；
- **FFTW**（同板性能，端到端）：自研 v4 版 5.143ms vs FFTW 版 8.154ms
  （同日同口径复测；9 月历史值 8.157ms 互相印证，两次堆值逐字节一致），
  **端到端快 36.9%**（真反超，远超测量噪声）；峰值堆
  **1.39MB 比 FFTW 版 1.84MB 省 24%**，且前向过程零 malloc。
  反超来自两层：v3 专用化集成（twiddle 表常驻 + 权重频谱加载期预计算，
  把每图 320 次冗余权重 FFT 全部前置）+ v4 codelet 特化。算子级仍未快过
  FFTW 标量 codelet 这一事实不变；若 FFTW 版同样预计算权重频谱并缓存 plan，
  预计仍会领先——但作为嵌入式专用实现，本组在端到端、内存与零动态分配上
  全面占优。详见任务一 README。

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
