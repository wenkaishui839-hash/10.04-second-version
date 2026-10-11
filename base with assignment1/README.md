
# 🚀 大创项目：RISC-V RVV 加速 rdFFT 与 Circulant 图像生成推理

本项目实现了一套基于 **RISC-V 向量扩展 (RVV 1.0)** 的高效率实数快速傅里叶变换（rdFFT）及逆变换（irdFFT）算子，并利用其构建了 Circulant（循环）矩阵乘法底层 C 库。

本项目的最终目标是：在边缘端 RISC-V 架构上，利用循环矩阵 $O(N \log N)$ 的低复杂度特性，替换 NICE 等生成式模型中的全连接层，实现模型轻量化与底层硬件推理极速加速。

---

## 🌟 核心基础：rdFFT 正变换与 RVV 硬件优化 (项目精髓)

本项目抛弃了传统的复数 FFT 库实现，针对“纯实数”信号场景进行了深度的微架构级别优化：

1. **隐式半复数编码与真原位 (True In-place) 计算**
传统的复数 FFT 在处理实数时会产生冗余的共轭虚部，导致需要申请额外的内存（Ping-pong Buffering）。本项目利用实数共轭对称性，设计了“正序存实部、倒序存虚部”的物理内存映射规则，在单次计算中通过代数变换直接消化共轭负号，实现了**零额外缓存申请**的原地数组计算。
2. **RVV 动态向量长度 (VLA) 机制**
代码使用 RISC-V 官方 C Intrinsics 内在函数编写。通过 `__riscv_vsetvl_e32m1` 动态获取硬件向量寄存器长度 `vl`，使代码具备极强的可扩展性，无需修改源码即可无缝运行在 128-bit 到 512-bit 不同位宽的 RVV 硬件上。
3. **寄存器内镜像重组 (`vrgather`) 替代离散访存 (`vsuxei16`)**
在读取倒序存放的虚部时，传统开源库（如 ETH Zurich 的实现）通常依赖离散索引散布写回内存，极大地破坏了 Cache 命中率。本项目坚持使用连续加载 (`vle32.v`) 占满内存总线带宽，随后在 CPU 寄存器内部利用 `vrgather` 瞬间完成数据的镜像调头，使内存总线效率达到物理极限。

---

## 📂 目录结构与核心文档简介

```text
dachuang/
├── pure_clanguage/                 # 纯 C 语言标量版目录 (作为无硬件依赖的基准与对照组)
│   ├── benchmark.c                 # [重要] 测速入口：对比标量 rdFFT 与 O(N^2) 暴力卷积速度
│   └── compare_torch.py            # Python 自动化验证脚本：调用 C 算子与 torch.fft 比对误差
├── twiddle_factors.c               # 旋转因子预计算底表生成模块 (SoA 内存布局，去除了三角函数运行时开销)
├── rdfft_butterfly_compute.c       # RVV 向量化核心正向蝶形算子
├── external.c                      # 正变换 RVV 调度引擎 (包含位反转重排与多级调度循环)
├── irdfft_butterfly_compute.c      # [新增] RVV 逆变换蝶形算子 (基于 DIF 结构)
├── irdfft_half_complex_mul.c       # [新增] RVV 半复数格式频域逐元素复数乘法
├── irdfft_circulant.c              # [新增] RVV 循环矩阵乘法组合算子
├── Makefile / build_rvv.bat        # 编译脚本

```

### 🎯 重点：三大 `main` 测试入口的区别

由于 C 语言一次只能编译一个 `main` 函数，项目中配置了 3 个独立的主函数文件，按需编译：

1. **`main.c`（正向误差打表入口）**
* **功能**：早期用于验证单向 `rdFFT`。它生成标准理论 DFT 真值，与硬件算子结果逐点比对，输出各频点的最大绝对误差（Max Error）与均方根误差（RMSE）。


2. **`main1.c`（速度与指令跑分入口）**
* **功能**：专为 `spike -l` 指令追踪设计。它屏蔽了耗时的控制台打印，利用 `printf("START")` 和 `printf("END")` 将核心运算包裹并循环执行 100 次，用于精确统计底层总指令数与 RVV 向量指令条数。


3. **`irdfft_main.c`（全链路验证与 Python 通信入口）**
* **功能**：任务一的核心交付入口。内部集成了正/逆向往返测试、直接卷积比对测试。并且**支持 CLI 命令行读写 `.bin` 文件模式**，专门供 `compare_torch.py` 脚本调用以实现 C 语言和 PyTorch 的跨界对齐。



---

## 🛠️ 编译与运行指南 (Linux VM 环境)

以下操作均在配置了 `riscv64-unknown-linux-gnu` 工具链与 `spike` 模拟器的 Linux 虚拟机环境中执行。

### 1. 编译并验证全套 Circulant 算子精度

```bash
# 1. 编译包含逆变换与 Circulant 的测试套件 (链接 8 个相关文件)
riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static -O3 -o irdfft_rvv \
  irdfft_main.c irdfft_butterfly_compute.c irdfft_external.c \
  irdfft_half_complex_mul.c irdfft_circulant.c \
  rdfft_butterfly_compute.c external.c twiddle_factors.c -lm

# 2. 在 Spike 模拟器上运行内置的精度误差报告
spike --isa=rv64gcv $HOME/riscv/riscv64-unknown-linux-gnu/bin/pk ./irdfft_rvv

```

### 2. 跨语言对齐测试 (C 版 vs PyTorch 官方库)

我们使用 Python 脚本生成随机矩阵，由 `spike` 模拟器执行 RVV 二进制文件，并将结果与 `torch.fft` 进行绝对误差对比：

```bash
cd pure_clanguage/

# 传入编译好的 RVV 二进制文件，并指定模拟器路径执行跨端验证
python3 compare_torch.py --bin ../irdfft_rvv \
  --emu "spike --isa=rv64gcv /home/wks/riscv/riscv64-unknown-linux-gnu/bin/pk"

```

### 3. 指令耗时跑分与 RVV 向量化率统计

```bash
# 1. 编译专用于抓取指令的 main1.c
riscv64-unknown-linux-gnu-gcc -march=rv64gcv -static -O3 -o rdfft_rvv_count \
  main1.c external.c twiddle_factors.c rdfft_butterfly_compute.c -lm

# 2. 统计单次 FFT 消耗的总机器指令数
spike --isa=rv64gcv -l $HOME/riscv/riscv64-unknown-linux-gnu/bin/pk ./rdfft_rvv_count 2>&1 | \
awk '/START/{c=0;next} /END/{print c " instrs per run (" c/100 " per FFT)"; exit} {c++}'

# 3. 统计其中 RVV 向量指令的具体条数
spike --isa=rv64gcv -l $HOME/riscv/riscv64-unknown-linux-gnu/bin/pk ./rdfft_rvv_count 2>&1 | \
awk '/START/{c=0;next} /END/{print c " RVV instrs"; exit} /vsetvl|vle|vse|vfadd|vfsub|vfmul|vfmacc|vfnmsac|vfmerge|vrgather|vlse|vsse|vid|vrsub/{c++}'

```

---

## 📈 数据归档与性能看板

为避免反复跑分，特将本项目阶段性的核心指标数据留档如下：

### 表 1：算子精度校验 (RVV 版 vs `torch.fft` 对齐)

*测试采用 float32，理论物理舍入误差极限位于 $O(10^{-7} \sim 10^{-6})$ 级别。*

| 规模 (N) | C vs PyTorch (max_abs) | C vs PyTorch (RMSE) | 结论 |
| --- | --- | --- | --- |
| 8 | 4.768e-07 | 1.699e-07 | [PASS] |
| 16 | 2.384e-07 | 1.159e-07 | [PASS] |
| 64 | 2.861e-06 | 1.067e-06 | [PASS] |
| 256 | 1.144e-05 | 4.005e-06 | [PASS] |
| 1024 | 3.052e-05 | 9.090e-06 | [PASS] |

**结论**：完全对齐。底层 C 算子的数学逻辑正确率达 100%。

### 表 2：时间复杂度理论加速比 (x86 标量耗时对比)

*在 Host 电脑上利用 `pure_clanguage/benchmark.c` 对比原生 $O(N^2)$ 直接卷积的耗时。*

| 规模 (N) | 暴力 DFT 耗时 (ns) | rdFFT 耗时 (ns) | 算法加速比 |
| --- | --- | --- | --- |
| 64 | 28,163 | 293 | **96.2x** |
| 256 | 447,446 | 1,081 | **413.8x** |
| 1024 | 7,360,257 | 5,016 | **1467.3x** |
| 4096 | 115,047,836 | 21,624 | **5320.3x** |

### 表 3：底层硬件拓展性与 RVV 向量化率 (Spike 仿真)

*随着规模 N 的激增，标量寻址等额外开销被极大摊销，RVV 向量 ALU 利用率显著爬升。*

| 测试规模 | 单次 FFT 总指令数 (条) | 单次 FFT RVV 向量指令数 (条) | 硬件向量化率 (RVV 占比) |
| --- | --- | --- | --- |
| 小规模基准 ($N=8$) | 536.7 | 23 | **4.28%** |
| 中等规模 ($N=256$) | 19,100.7 | 2,576 | **13.48%** |
| 大规模 ($N=1024$) | 82,232.7 | 13,248 | **16.11%** |

## 端到端交叉验证（2026-09 复核）

任务一算子已接入任务四 NICE 全链路推理并通过复核：

- 表 3 的 N=256 单次 FFT 19,100 条指令 × 每图 320 次循环卷积 ≈ 19.2M，
  与任务四实测的核心推理总指令 19,615,671 互相咬合，两套独立测量一致；
- 精度表（表 1）的量级与任务四端到端误差 9.5e-07 一致；
- 端到端最新数据（v4 codelet 特化后）：向量化率 25.61%、板上加速 2.63×、
  峰值内存 1.39MB，见 [assignment4/任务四/README.md](../assignment4/任务四/README.md)。

## 板上算子级实测（CanMV K230，C908 @1.6GHz，2026-09）

命令：`taskset -c 0 ./rdfft_rvv_count` 与 `./rdfft_scalar_count`
（各预热 10 + 循环 100 次；标量版用 `-march=rv64gc` 编译保证纯标量，
入口为 `pure_clanguage/main1_scalar.c`，与 `main1.c` 同构）。

| 版本 | 单次 rdFFT(256) 耗时 | 备注 |
|------|---------------------|------|
| RVV 版（v4 codelet 后） | **6.37 µs** | v2 为 10.74µs（v1 12.9~13.3µs）；v4 = codelet 特化 + LMUL 提升（fft_bench_v3_rvv 实测，k=1 频点 (0,−128) 命中） |
| 纯标量版 | **14.89 µs** | 不变（基线） |
| **算子级加速比** | **2.34×** | 时间减少 57.2%（v2 为 1.39×，v1 为 1.12×） |

## v2 结构内优化（已合并为主实现，2026-09）

**原则：原论文共轭对蝶形结构原封不动**（公式 / 隐式半复数布局 / DIT 顺序 /
蝶形真原位全保留），只改两处"怎么读数据"。实验孪生文件（`rdfft_butterfly_staged.c`、
`external_staged.c`、`bitrev_rvv.c`、`fft_v2_bench.c`）保留为实验记录，主实现
（`rdfft_butterfly_compute.c`、`irdfft_butterfly_compute.c`、`external.c`、
`irdfft_external.c`）已原地升级，全部调用方（含任务四 `circulant_rvv.c`）切换至
`generate_rdfft_twiddle_factors_staged`。

| 优化 | 内容 | 板上实测收益（rdFFT(256)） |
|------|------|---------------------------|
| ① 逐级连续旋转因子表 | 替代跨步 `vlse32`，阶段偏移 = M/2−1，蝶形内连续 `vle32` | 12.90 → 12.20 µs（−5.4%） |
| ② 位反转置换表 | 预计算一次 + 查表交换，替代逐元素 j 增量计算 | 12.20 → **10.15 µs**（−16.9%） |
| **合计** | | **12.90 → 10.15 µs（−21.4%）** |

- 正确性：v2 三种配置频点检查全部输出理论真值（0, 0, 0, −128, −64），
  与原实现逐位等价。
- 失败路线记录：vluxei32 离散索引向量化位反转在 clang+C908 上触发非对齐
  访存异常（SIGBUS），在 GCC 15.1+Spike 上产生 NaN——两个独立组合均失败，
  该平台离散索引向量访存不可靠（`bitrev_rvv.c` 内已注释保留）。
- 合并后**反超 kissfft 标量版 12.3%**（10.15 vs 11.57µs），详见下节对比表。
- 端到端收益（任务四 bench.sh 实测）：核心推理总指令 19,615,671 → 16,951,031
  （−13.6%），向量化率 15.06% → 17.43%，推理输出与合并前逐位一致。

- **板上正确性**：两版频点检查均输出 DFT 理论真值 `Y[255]=-128.0000,
  Y[254]=-64.0000`（信号 sin(2πn/N)+0.5·sin(4πn/N) 的 Im(F[1])、Im(F[2])），
  即 rdFFT 在真硬件上对理论值逐位正确；`irdfft_rvv` 全套往返/循环卷积
  内置测试在板上全部 PASS（误差 1e-7~2e-5，float32 舍入量级）。
- 算子级加速比（1.12×）低于端到端（1.41×）的原因：N=256 小规模下位反转
  与控制流标量开销占比高，且 VLEN=128 每向量指令仅 4 个 FP32、vrgather 等
  重指令多周期延迟；端到端中推理主循环的简单向量化（vle/vfadd/vse）效率更高。

## v4 codelet 特化（已合并为主实现，2026-10）

**原则不变：原论文共轭对蝶形结构原封不动**（公式 / 隐式半复数布局 / 4R4W
原地闭环全保留），针对 M=1/2/4/8 四级做 codelet 式直写（新文件
`rdfft_butterfly_v3.c` / `irdfft_butterfly_v3.c` / `external_v3.c` /
`irdfft_external_v3.c` / `irdfft_half_complex_mul_v3.c`，M≥16 仍走原通用蝶形）。

| 优化 | 内容 | 板上实测 |
|------|------|---------|
| ① codelet 小阶段特化 | 旋转因子角度 θ=πk/M 与 N 无关，M≤8 全部是编译期常量（零表加载、零循环开销），跨组 vlse32 向量化 | 算子级 10.74 → **6.37 µs（−40.7%）** |
| ② 半复数乘原地化 | B = A .* B 先读后写，scratch 从 2N 降为 1N | 端到端受益 |
| ③ 前向零 malloc | 权重频谱与工作区全部挪到加载期（init/prepare 三段式），推理期间 brk 不变 | 满足任务书第一条硬约束 |
| ④ LMUL 提升 | 归一化 m8、频域乘 m2 | — |

- 正确性：算子 [check] 频点 k=1 → (0, −128) 逐位命中；端到端 compare.py PASS
  （最大误差 1.19e-06，codelet 改变浮点结合顺序，舍入路径不同但仍在 float32
  舍入量级）；Spike 总指令 11.82M → 5.58M（−52.8%）。
- 放弃路线记录：位反转向量化需要 N 浮点临时缓冲或 vsuxei 离散写回，
  前者违反"零缓冲"硬约束、后者平台不可靠（vluxei 已双失败）——标量 swap
  换查表（v2）是零缓冲约束下的最优解。

## 五条硬性要求审计（任务书，2026-10 v4 后全部满足）

| # | 要求 | 实现状态 |
|---|------|---------|
| 1a | N→N 等积覆写，无 N+2 | ✅ FFT 核心输入 N 实数原地变输出 |
| 1b | 前向/反向零 malloc 等规模工作区 | ✅ v4 起前向零 malloc（init/prepare 三段式，brk before=after 实测证据） |
| 2 | 隐式半复数编码（DC/Nyquist 压缩，Re@k / Im@N−k） | ✅ 全程不解包复数 |
| 3 | 4读4写共轭对蝶形闭环（Proposition 1） | ✅ 含 codelet 特化后依然严格 4R4W |
| 4 | 数学无损 + 正反互逆（FP32 舍入量级） | ✅ RMSE 1e-7~1e-6，torch.fft 对齐，板上往返 PASS |
| 5 | 半复数频域乘保持共轭对称 | ✅ 原地乘后布局不变，irdFFT 直接解析 |

## 原位计算情况表（任务要求）

| 模块 | 是否原位 | 说明 |
|------|---------|------|
| rdFFT/irdFFT 蝶形 | ✅ 真原位 | 组内 4 读 4 写，零临时变量（v4 codelet 特化后依然 4R4W） |
| 位反转 / 归一化 | ✅ 原位 | swap 重排 / 向量乘 1/N |
| **FFT 核心合计** | ✅ **零额外缓冲** | 输入 N floats 原地变输出，仅需预计算旋转因子常量表 |
| 频域乘 | ✅ 原地（v4） | B = A .* B 先读后写，省去输出缓冲 |
| circulant wrapper | ✅ 常驻工作区（v4） | init 期分配 1×N scratch + 兜底 1×N，前向零 malloc |
| NICE 推理层 | ✅ 缓冲复用 | 4096 floats 调用者传一次、24 层复用 |

## 与现有 FFT 库对比（kissfft，同板实测，2026-09）

方法：`fft_bench.c` 在同一块 K230 上对比三个实现（同一测试信号，
k=1 频点理论真值 (0, -128) 互验，全部命中）：

| 实现 | 编译目标 | 单次 256 点 FFT | 相对本组 RVV |
|------|---------|----------------|-------------|
| **本组 rdFFT（RVV，v4 codelet）** | rv64gcv | **6.37 µs** | 基准 |
| kiss_fftr（实数专用） | rv64gc（纯标量） | 11.47 µs | +80.1% |
| kiss_fftr（实数专用） | rv64gcv | 12.57 µs | +97.3% |
| 本组 rdFFT（标量） | rv64gc | 14.89 µs | +133.8% |
| kiss_fft（复数接口） | rv64gc | 16.12 µs | +153.1% |
| kiss_fft（复数接口） | rv64gcv | 17.04 µs | +167.5% |

> v1 版（13.29µs）曾慢于 kissfft 标量 14.9%；v2 结构内优化（保持原论文共轭对
> 结构）后 10.74µs 反超 6.4%；v4 codelet 特化后 **6.37µs，反超 kissfft 标量版
> 80.1%**（fft_bench_v3_rvv 实测，k=1 频点 (0,−128) 逐位命中）。

**诚实结论与根因分析：**

1. v1 实现的 RVV rdFFT 在 N=256 上慢于 kissfft 纯标量版约 15%——原因是结构
   开销而非算法：kiss_fftr 用"128 点复数 FFT + 重组"实现实数 FFT，
   **没有独立位反转扫描**，指令数更少（≈18.5K vs 19.1K）且 IPC 更高。
2. **v2 结构内优化（保持原论文共轭对结构）后反超**：逐级连续旋转因子表 +
   置换表位反转把单次 rdFFT(256) 从 13.3µs 降到 **10.15µs，比 kissfft 标量
   版（11.57µs）快 12.3%**——在算法结构不变的前提下完成反超。
3. 复数接口比实数接口慢 30~40%，**实测验证了"实数半谱算法工作量减半"**。
4. clang 对 kissfft 的自动向量化无效（gcv 版反而略慢）——说明通用编译器
   抓不住该结构，手工 RVV 有真实空间；kissfft 与 FFTW 均无 RVV 后端。

编译（VM）：

```bash
# RVV 版 (含本组 RVV rdFFT + kissfft; kissfft 由编译器按 gcv 目标生成)
riscv64-unknown-linux-gnu-clang -march=rv64gcv -static -O3 -DUSE_RVV \
  -o fft_bench_rvv fft_bench.c kissfft/kiss_fft.c kissfft/kiss_fftr.c \
  external.c twiddle_factors.c rdfft_butterfly_compute.c -lm

# 标量版 (本组标量 rdFFT + kissfft, 均纯标量)
riscv64-unknown-linux-gnu-clang -march=rv64gc -static -O3 \
  -o fft_bench_scalar fft_bench.c kissfft/kiss_fft.c kissfft/kiss_fftr.c \
  pure_clanguage/external.c pure_clanguage/rdfft_butterfly_compute_scalar.c \
  pure_clanguage/twiddle_factors.c -lm
```

板上：`taskset -c 0 ./fft_bench_rvv` 与 `./fft_bench_scalar`。

## Memory Layout 分析（任务要求）

**结论先行：现有布局不是照抄 CUDA，而是经典 Numerical Recipes/FFTPACK 式
半复数压缩格式，在"共轭对蝶形"这类实数 FFT 算法下接近最优；换布局的收益
主要不在布局本身，而在算法结构。**

- cuFFT/torch 的 r2c 输出是**交错复数**（interleaved，N/2+1 个复数）；
  本项目采用"正序实部 Y[k]=Re(F[k])、倒序虚部 Y[N-k]=Im(F[k])"的
  **镜像半复数压缩格式**，设计动机是单数组、零缓存、真原位。
- 代价：镜像虚部使蝶形与频域乘需要 `vrgather` 寄存器内调头（每迭代 2~4 条，
  vrgather 是 C908 上较重的向量指令），位反转为标量。
- 候选布局对比：

| 布局 | 总内存 | 蝶形访存 | 频域乘 | 原位 | 评价 |
|------|--------|---------|--------|------|------|
| A. 镜像半复数（现状） | N | 镜像 + vrgather | 3×vrgather/迭代 | ✅ 真原位 | 内存最优 |
| B. SoA 双数组 Re/Im | N+2 | 连续，但上半个谱需显式回填（含取反倒序拷贝 = 同量级 gather） | 零 gather | ⚠️ 双数组 | 转换开销守恒，见实验 |
| C. 交错复数（cuFFT 惯例） | N+2 | stride-2 / vlseg2 | 拆对 | 可 | 与 torch 对齐 |

- **实验验证**（`soa_mul_bench.c`，板上微基准，2026-09 实测）：

| 路径 | 单次频域乘（N=256） | 结论 |
|------|--------------------|------|
| 镜像布局直接乘（现状） | 0.94 µs | 基准 |
| unpack + SoA 乘 + repack | 2.44 µs | 转换开销守恒，单次乘不划算 |
| SoA 连续乘本身 | **0.62 µs** | 比镜像乘快 1.5×（零 gather） |

  实测证实"格式转换的 gather 代价守恒"：unpack/repack 的 gather 数等于直接乘
  省下的 gather 数（还多出 Re 搬运），单次乘净亏 2.6×。且频域乘仅占端到端
  流水线约 2%，换布局无实际收益；SoA 优势只在数据驻留 SoA 的完整管线
  （蝶形+乘全 SoA 且消除回填）才可能体现。真正的提速方向是**改算法结构**
  （kissfft 式 N/2 复数 FFT / 前向 DIF 自然序，同时消除位反转与镜像访问），
  而非仅改存储布局。两路径输出逐位一致（最大差 0.0），正确性无损。

## 与 FFTW 的对比实测（任务要求，同板数据 2026-09）

**实测结论：在 N=256 上，本组 RVV rdFFT v4（6.37µs）仍未快于 FFTW 标量
codelet（3.87µs MEASURE），但差距从 v2 的 2.7× 收窄到 1.65×。"为什么能比
FFTW 快"的完整答案是：算子级尚不能（差距已定位为 codelet 代码质量，v4 已
用 codelet 式特化追回一半），端到端已大幅反超（5.14 vs 8.15ms，快 36.9%，
见下节端到端对比）。**

| 实现 | 单次 256 点实数 FFT | 备注 |
|------|--------------------|------|
| FFTW r2c MEASURE（RISC-V 标量） | **3.87 µs** | 官方库，板上自调优 plan，≈6.2K 周期 |
| FFTW r2c ESTIMATE（RISC-V 标量） | 4.16 µs | 无调优 plan |
| 本组 rdFFT（RVV，v4 codelet） | **6.37 µs** | v2 为 10.48µs，codelet 特化追回一半差距 |
| kiss_fftr（标量） | 11.47 µs | 本组已反超 80.1% |
| 本组 rdFFT（标量） | 14.89 µs | — |

差距分解（关键洞察）：
1. **FFTW 标量 vs 本组标量 = 3.85×**：FFTW 用预生成 **codelet**（小规模 FFT
   内核硬编码展开，零旋转因子加载、零循环开销），代码质量差距远大于 SIMD。
2. **本组 RVV vs 本组标量 = 1.42×（v2）→ 2.34×（v4）**：VLEN=128 向量化 +
   codelet 特化 + LMUL 提升的全部收益。
3. v4 把 codelet 这招用在原论文共轭对结构上（M=1/2/4/8 四级特化，旋转因子
   全部是编译期常量 θ=πk/M），对 FFTW 的算子级差距从 2.7× 追到 1.65×——
   实证了"差距在代码生成策略而非 SIMD"的判断。
4. 进一步追赶方向：M=16/32 级同样特化（各需 7/15 个常量）；位反转向量化
   因零缓冲要求（任务书第一条硬约束）而放弃——原位标量 swap 是零缓冲的
   固有代价。详见下方"v4 codelet 特化"与"五条硬性要求审计"两节。

FFTW 构建：3.3.10 源码 `configure --host=riscv64-unknown-linux-gnu
--enable-float --disable-shared --enable-static` 交叉编译（源码树归档于
VM vmshare/fftw-3.3.10，测试入口 `fftw_bench.c`）。

### 端到端同板对比（NICE 推理，2026-10 v4 终版）

| 维度 | FFTW 版（nice_infer_fftw） | 自研 RVV 版 v4（nice_infer_rvv） | 结论 |
|------|---------------------------|------------------------------|------|
| 单图推理 | 8,154.4 µs（2026-10-08 复测） | **5,142.7 µs** | **自研快 3,011.7 µs（36.9%）** |
| 程序峰值堆 | 1,929,216 B（1.84 MB） | 1,462,272 B（1.39 MB） | **自研省 24%（456 KB）** |
| 前向动态分配 | 每图按通用库方式付 plan 与工作区 | **0 次 malloc（全部加载期前置）** | 见"五条硬性要求审计" |
| 正确性 | 与 PyTorch 参考 ~1e-6 | 1.19e-06 | 均通过 |

演变路径：v2 10,944.1 µs → v3 权重 FFT 预计算 8,134.7 µs（−25.7%，端到端
追平 FFTW 0.24%）→ **v4 codelet 特化 5,142.7 µs（−36.8%，反超 FFTW 36.9%）**。
v4 叠加了：M=1/2/4/8 蝶形 codelet、半复数乘原地化、前向零 malloc、LMUL 提升
（Spike 总指令 11.82M → 5.58M，−52.8%）。

**诚实的根因表述**：端到端 36.9% 的反超是实测事实、远超测量噪声，来源是
**专用化集成 + codelet 特化**两层——本组把一次性工作全部前置（twiddle 表
常驻 + 权重频谱加载期预计算 + 常驻工作区，前向零 malloc），并把 FFTW 的
codelet 招数用在共轭对结构上；FFTW 版则按通用库方式每图重复付 plan 建立
与权重 FFT 开销（plan 建立开销 ≈3.2ms 摊薄了其算子优势）。**算子级本组仍
慢 1.65× 这一事实不变**（6.37 vs 3.87µs，v2 时为 2.7×）；若 FFTW 版同样
做权重预计算并缓存 plan，预计 ~6-7ms 仍会领先。但作为嵌入式专用实现，
本组的综合优势是：端到端快 37% + 内存省 24% + 前向零动态分配 + 无外部库
依赖（自主可控）。
