# 任务四: C 语言 / Spike 端到端推理与 RVV 跑分

目标: 用纯 C 手写 `PaddedBlockCirculantNICE` 的**生成方向前向推理**（Z → X），
在 Spike 模拟器上运行并统计 RVV 指令占比。

## 任务分解（全部完成）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | 权重读取（`fopen`/`fread` 二进制） | ✅ `load_weights.c` + `convert_weights.py` |
| 2 | 手写前向推理主逻辑（切分、Circulant 算子、ReLU、ScalingLayer、裁剪） | ✅ `inference.c`（与 PyTorch 比对通过） |
| 3 | Spike 仿真跑分 + RVV 指令统计 | ✅ `circulant_rvv.c` + 任务一 RVV 算子 |

## 文件结构

| 文件 | 作用 |
|------|------|
| `convert_weights.py` | 把任务三导出的文本权重 `.txt` 转成二进制 `.bin`（只跑一次） |
| `load_weights.h/.c` | 第一步: `fopen`/`fread` 读取两个 `.bin` 到 C 内存 |
| `circulant_op.h` | 循环卷积算子统一接口（`circulant_op_init` / `circulant_op_f32`） |
| `circulant_scalar.c` | 标量 FFT 算子（x86 校验 + RISC-V 标量基线共用） |
| `circulant_rvv.c` | RVV 算子封装（Spike/板子跑分用，链接任务一算子） |
| `inference.h/.c` | 第二步: NICE 生成方向前向推理主逻辑（手工 RVV 向量化版） |
| `inference_scalar.c` | 推理主逻辑纯标量孪生文件（bench.sh 标量基线用，与 inference.c 数学一致） |
| `main.c` | 第一步自检程序（`load_weights_test`） |
| `infer_main.c` | 推理入口：START/END 计数标记、brk 堆测量、reps 参数与微秒计时（结果写 stdout） |
| `gen_reference.py` | 生成固定 Z + PyTorch 参考输出（虚拟机 torch 环境跑） |
| `compare.py` | 比对 C 输出与 PyTorch 参考（纯 Python） |
| `bench.sh` | 一键跑分：编译标量/RVV 两版、spike 指令统计、比对、堆内存、文件大小 |
| `z_input.txt` / `x_ref.txt` | 固定测试输入（1024 维）与 PyTorch 参考输出（784 维），比对基准 |
| `Makefile` | 编译 / 运行脚本（早期流程；现以 `bench.sh` 为主） |

RVV 版额外链接任务一的 7 个算子源文件（归档于仓库根目录 `base with assignment1/` 下）:
`irdfft_butterfly_compute.c`、`irdfft_external.c`、`irdfft_half_complex_mul.c`、
`irdfft_circulant.c`、`rdfft_butterfly_compute.c`、`external.c`、`twiddle_factors.c`。

## 运行流程

```bash
cd ~/桌面/dachuang/任务四

make convert        # ① 一次性: 文本权重 .txt -> 二进制 .bin
make run-load       # ② 第一步自检
make run-infer      # ③ 第二步: 标量推理 (x86), 输出 x_out.txt
python3 compare.py  # ④ 比对标量推理结果
make run-infer-rvv  # ⑤ 第三步: RVV 推理 (Spike), 输出 x_out_rvv.txt
python3 compare.py x_out_rvv.txt   # ⑥ 比对 RVV 推理结果
```

## 验证结果

### 第一步: 权重读取

`make run-load` 预期输出:

```
block-circulant 层数 : 24
总参数量            : 105472 (期望 105472)
权重数值和 (校验用) : 1880.16069
[自检通过] 层数与参数量与 PaddedBlockCirculantNICE 一致。
```

### 第二步: 正确性比对

`python3 compare.py x_out_rvv.txt` 实测: `最大绝对误差 1.19e-06`、`RMSE 2.01e-07`、
`结论: 通过`。这说明 NICE 切分 / 层序 / reverse 标志 / ScalingLayer / 裁剪全部正确，
且 RVV 推理与 PyTorch 参考在 float32 舍入量级内一致。

### 第三步: RVV 指令统计

| 指标 | 优化前（文本加载） | 优化后（二进制加载） |
|------|-------------------|---------------------|
| 总指令数 | 79,330,206 | 28,548,887 |
| RVV 向量指令数 | 3,572,712 | 3,376,156 |
| 向量化率 | 4.50% | **11.83%** |

> RVV 版最终保留的优化是**二进制权重加载**（`load_weights.c` 由文本 `strtod`
> 改为二进制 `fread`），总指令数下降约 64%，向量化率从 4.50% 提升到 11.83%。
> 曾尝试的「位反转向量化」（`external.c` / `irdfft_external.c` 用 `vluxei32` gather）
> 在 GCC 15.1.0 / Spike 上产生 NaN（每个 256 块前 64 个元素出错），已回退为标量 swap，未保留。

## 最终效果

**① `make run-load` 输出（关键行）：**

```
block-circulant 层数 : 24
ScalingLayer 特征数  : 1024
总参数量            : 105472 (期望 105472)
权重数值和 (校验用) : 1880.16069
[自检通过] 层数与参数量与 PaddedBlockCirculantNICE 一致。
```

**② `python3 compare.py x_out_rvv.txt` 输出：**

```
维度: 784
最大绝对误差: 1.192000e-06  (下标 329, C=0.980170846, PyTorch=0.980172038)
均方根误差(RMSE): 2.014592e-07
结论: 通过 —— 误差在 float32 舍入量级, C 推理与 PyTorch 一致
```

**③ RVV 指令统计（`spike -l` 日志）：**

```
总指令数       : 28,548,887
RVV 向量指令数 : 3,376,156
向量化率       : 11.83%
```

## 本版跑分方法（速度 / 内存对比，bench.sh）

老师要求的"速度与内存节约"统一由 `bench.sh` 一键测量（脚本头有完整说明）：

```bash
chmod +x bench.sh
./bench.sh
```

| 测量项 | 方法 | 口径说明 |
|--------|------|---------|
| 核心推理指令数 | `spike -l` 轨迹，只统计 START/END 区间 | 权重加载 / 输出打印均在区间外；标量基线用 `-march=rv64gc`（无 V）编译保证纯标量，推理主逻辑链接纯标量孪生文件 `inference_scalar.c`；RVV 指令用 `v` 前缀 mnemonic 正则统计（补齐早期漏掉的 `vid.v` 等） |
| 程序堆内存 | `infer_main.c` 内 `brk(0)` 原始系统调用打印（stderr） | C 程序真实堆占用（实测 ≈1.15 MB）；勿用 `/usr/bin/time` 的 Max RSS 当程序内存，其中 ~12 MB 主要是 Spike 模拟器本身 |
| 权重内存 | 两个 `.bin` 文件大小 | 与原版 NICE 19.15M 参数 ≈ 76.6 MB 对比，压缩率 ≈ 99.44% |
| 正确性 | 两版输出均过 `compare.py` | 标量版也通过比对后，其指令数才可信 |

## 实测结果（bench.sh 回填，2026-09）

| 指标 | RVV 版 | 纯标量版 | 说明 |
|------|--------|---------|------|
| 核心推理总指令 | 19,615,671 | 31,227,060 | RVV 版为标量版的 62.8%（减少 37.2%） |
| RVV 向量指令 | 2,953,760 | 0 | — |
| 向量化率 | 15.06% | 0.00% | 含 FFT 算子 + 推理主循环向量化 + 编译器自动向量化 |
| 输出正确性 | 最大误差 9.50e-07 | 最大误差 7.75e-07 | 均通过 compare.py（float32 舍入量级） |
| 程序峰值堆内存 | 1,179,648 B（≈1.15 MB） | 同左 | brk(0) 实测，不含 Spike 模拟器本身 |
| 权重文件大小 | 418,280 + 8,200 = 426,480 B | — | vs 原版 NICE ≈ 76.6 MB，压缩率 ≈ 99.44% |

口径说明：
- 早期手动统计（19,621,692 / 2,836,003 / 14.45%）与本次差异来自：统计正则补齐了
  早前漏计的 `vid.v` 等整数向量指令（约 12 万条），以及 infer_main 预热/循环包装的
  细微代码差异，测量口径一致。
- heap 的 before 与 after 相等：预热推理已把 brk 推到峰值，两者都是峰值堆占用。

## 板上实测（CanMV K230，C908 @1.6GHz，RVV 1.0，2026-09）

命令：`taskset -c 0 ./nice_infer_* circulant_weights.bin scaling_weights.bin z_input.txt 100`

| 指标 | RVV 版 | 纯标量版 | 说明 |
|------|--------|---------|------|
| 单图推理耗时 | 13,481 µs（3 次平均：13.63 / 13.54 / 13.28 ms） | 19,041.7 µs | **实测加速比 1.41×**（时间减少 29.2%） |
| 峰值堆内存 | 1,179,648 B（≈1.15 MB） | 同左 | 板上实测，与 Spike 端一致 |
| 输出正确性 | 与 PyTorch 参考一致（9.50e-07） | 同左 | 板 = VM = PyTorch 三方一致 |

解读：指令数减少 37.2%，实测时间加速 29.2%，差距来自 VLEN=128（每向量指令仅 4 个
FP32）与 C908 顺序核上 vrgather/vlse/vfdiv 的多周期延迟；向量化率 15% 意味着 85%
时间仍在标量部分（位反转、控制流、每算子 malloc/free、冗余权重 FFT），是后续优化空间。

### 板载输出正式比对

板上生成的 `x_out_board.txt`（784 维，归档于仓库根目录）与 VM 端 `x_out_rvv.txt`
逐位一致（下标 494 处值 `1.07476306` 与 VM compare.py 报告完全一致），即
板 = VM = PyTorch（最大误差 9.5e-07），三方链路验证完成。

### 部署权重归档与校验（2026-09 复核）

- 部署模型为 `padded-block-circ`（PaddedBlockCirculantNICE，1024 维），其训练产物
  归档于 `assignment2 and 3/runs/padded_block_circ_nice_e50/`：
  `padded-block-circ_nice.pt`（checkpoint）、`circulant_weights.txt/.pt/.bin`、
  `scaling_weights.txt/.pt/.bin`、`metrics.json`（param_count=105,472）、样例图。
- 权重文件校验和 **1880.16069**，与 VM 端 `load_weights_test` 报告逐位一致（已复核）；
  `z_input.txt`（1024 维）与 `x_ref.txt`（784 维）归档于本目录，`x_ref.txt` 前 3 个
  值与板上 C 推理输出吻合至第 7 位小数，与 compare.py 的 9.5e-07 结论互为印证。

### 稳定性与测量条件（重要）

- 第二次板上跑分 15.6ms/图 被板载 WiFi 驱动（8189fs）的内核警告污染：警告栈
  （rtw_lps_state_chk）在计时区间内打印于 CPU0 串口控制台，占用被测核时间。
  干净复测方法：`ip link set wlan0 down` 后再测。
- 主频确认：内核无 cpufreq（固定主频）。`clk_summary` 的 `cpu0_src=800MHz` 是时钟
  树中间节点而非核时钟；由指令吞吐反推：标量版 31.2M 指令/19.0ms，若主频为 800MHz
  则 IPC≈2.05，超过 C908 双发射上限 2.0，故实际主频为 1.6GHz（此时标量 IPC≈1.03、
  RVV IPC≈0.91，符合双发射顺序核特征）。
- 最终定稿值（关 WiFi 干净环境）：RVV 13.5ms（3 次平均，波动 ±1.3%）/ 标量 19.0ms，
  加速比 1.41×。

## 改进历程（本阶段做了什么）

| # | 改进 | 效果 |
|---|------|------|
| 1 | 权重加载文本 strtod → 二进制 fread | 加载阶段总指令 79.3M → 28.5M |
| 2 | 推理主循环手工 RVV 化（relu/acc/bias/sub/div 五算子） | 端到端向量化率 11.83% → 15.06% |
| 3 | 跑分口径修复（正则补计 vid.v 等整数向量指令） | RVV 计数 +约 12 万条，口径更严谨 |
| 4 | 测量装置：bench.sh（标量基线 -march=rv64gc、START/END 区间计数、brk 堆测量、板上计时 reps 参数） | 得到标量 vs RVV 全量对比与真实内存数据 |
| 5 | infer_main 计时/堆测量改用原始 ecall 系统调用 | 修复 pk 加载器 ELF 段布局问题（"Access exception while loading payload"） |
