#!/usr/bin/env bash
# ======================================================================
# bench.sh — 本版跑分: RVV 版 vs 纯标量版 (同一份推理代码, 均在 Spike 上)
#
# 测量 5 项 (对应老师要求的速度与内存):
#   1) 核心推理总指令数:  spike -l 轨迹, 只数 START/END 区间 (权重加载在区间外)
#   2) RVV 向量指令数 / 向量化率 (v 前缀 mnemonic 全量统计, 含 vid.v 等)
#   3) 程序堆内存:        infer_main.c 内 sbrk(0) 打印 (不含 Spike 模拟器开销)
#   4) 权重文件大小:      两个 .bin 与原版 NICE 19.15M 参数 ≈ 76.6 MB 对比
#   5) 正确性:            两版输出与 PyTorch 参考比对 (compare.py)
#
# 公平性:
#   - 标量基线用 -march=rv64gc (无 V) 编译 => 编译器不会自动向量化
#   - 两版除 circulant 算子源文件外完全相同, 同编译器同 -O3
#
# 用法 (在 VM 的任务四目录, 已放好 *.bin 与 z_input.txt / x_ref.txt):
#   chmod +x bench.sh && ./bench.sh
#
# 注意: 第 2 步 spike -l 会输出上千万行指令轨迹, 需几分钟, 属正常。
# ======================================================================
set -e

# ---- 按你 VM 的实际环境修改这几行 ----
CC=riscv64-unknown-linux-gnu-clang
SPIKE=spike
PK=$HOME/riscv/riscv64-unknown-linux-gnu/bin/pk
# ---------------------------------------

# 推理主逻辑: VM 上的 inference.c 已手工 RVV 化 (relu/累加等), 无法用无 V 扩展的
# -march=rv64gc 编译; 标量基线改用其纯标量孪生文件 inference_scalar.c。
RVV_SRCS="circulant_rvv.c twiddle_factors.c rdfft_butterfly_compute.c \
          irdfft_butterfly_compute.c irdfft_half_complex_mul.c \
          irdfft_circulant.c irdfft_external.c external.c"
SCALAR_SRCS="circulant_scalar.c"

echo "== [1/5] 编译 (同编译器同 -O3, 标量版无 V 扩展) =="
$CC -march=rv64gcv -static -O3 -o nice_infer_rvv \
    load_weights.c inference.c        infer_main.c $RVV_SRCS    -lm
$CC -march=rv64gc  -static -O3 -o nice_infer_scalar \
    load_weights.c inference_scalar.c infer_main.c $SCALAR_SRCS -lm

count_instrs() {
  local bin=$1 isa=$2
  local res
  # stdout 与 stderr 全部合并进管道 (兼容标记打在任一流的 infer_main.c 版本;
  # 784 个输出数字在 END 之后, awk 已 exit, 不会混入计数)
  res=$($SPIKE --isa=$isa -l $PK ./$bin \
      circulant_weights.bin scaling_weights.bin z_input.txt \
      2>&1 | awk -v name="$bin" '
    /START/ { flag=1; total=0; rvv=0; next }
    /END/   { printf "[%s] 核心推理总指令 %d | RVV 向量指令 %d | 向量化率 %.2f%%\n",
                     name, total, rvv, (total ? rvv*100.0/total : 0.0);
              flag=0; exit }
    flag    { total++;
              if ($0 ~ / vset/ || $0 ~ / v[a-z0-9]*\.(v|f|s)/) rvv++ }')
  if [ -z "$res" ]; then
    echo "[$bin] 错误: 未捕获到 START/END 标记 (程序可能未正常跑完, 请手动运行一次看输出)" >&2
    return 1
  fi
  printf '%s\n' "$res"
}

echo "== [2/5] 指令统计 (START/END 区间, 单张图一次推理) =="
RVV_RES=$(count_instrs nice_infer_rvv    rv64gcv)
SCALAR_RES=$(count_instrs nice_infer_scalar rv64gc)
echo "$RVV_RES"
echo "$SCALAR_RES"
r_total=$(echo "$RVV_RES"    | sed -n 's/.*总指令 \([0-9]*\).*/\1/p')
s_total=$(echo "$SCALAR_RES" | sed -n 's/.*总指令 \([0-9]*\).*/\1/p')
echo "指令数对比: RVV 版为标量版的 $(awk -v a="$r_total" -v b="$s_total" 'BEGIN{printf "%.1f%%", a*100/b}')" \
     "(减少 $(awk -v a="$r_total" -v b="$s_total" 'BEGIN{printf "%.1f%%", 100-a*100/b}'))"

echo "== [3/5] 结果落盘并与 PyTorch 参考比对 =="
# pk 会把程序的 stdout+stderr 合并进 spike 的 stdout, 这里滤掉标记行
# ([heap]/START/END/[time]), 只留纯数值给 compare.py
$SPIKE --isa=rv64gcv $PK ./nice_infer_rvv    circulant_weights.bin scaling_weights.bin z_input.txt \
    2>/dev/null | grep -v -E 'START|END|\[heap\]|\[time\]' > x_out_rvv.txt
$SPIKE --isa=rv64gc  $PK ./nice_infer_scalar circulant_weights.bin scaling_weights.bin z_input.txt \
    2>/dev/null | grep -v -E 'START|END|\[heap\]|\[time\]' > x_out_scalar.txt
if command -v python3 >/dev/null 2>&1 && [ -f compare.py ]; then
  python3 compare.py x_out_rvv.txt    || echo "[警告] RVV 版比对失败 (x_out_rvv.txt 异常?)"
  python3 compare.py x_out_scalar.txt || echo "[警告] 标量版比对失败 (x_out_scalar.txt 异常?)"
else
  echo "(跳过: 无 python3 或 compare.py)"
fi

echo "== [4/5] 程序堆内存 (sbrk, C 程序真实堆占用, 不含模拟器) =="
$SPIKE --isa=rv64gcv $PK ./nice_infer_rvv    circulant_weights.bin scaling_weights.bin z_input.txt >heap_rvv.log    2>&1
$SPIKE --isa=rv64gc  $PK ./nice_infer_scalar circulant_weights.bin scaling_weights.bin z_input.txt >heap_scalar.log 2>&1
heap_rvv=$(grep heap heap_rvv.log       | tail -2 | tr '\n' ' ' || true)
heap_scalar=$(grep heap heap_scalar.log | tail -2 | tr '\n' ' ' || true)
echo "[RVV  ] ${heap_rvv:-（无 heap 输出 — 请同步最新 infer_main.c）}"
echo "[标量] ${heap_scalar:-（无 heap 输出 — 请同步最新 infer_main.c）}"

echo "== [5/5] 权重文件大小 (模型存储内存) =="
ls -l circulant_weights.bin scaling_weights.bin | awk '{printf "  %-28s %10d bytes\n", $9, $5}'
echo "  参考: 原版 NICE 19,150,000 参数 x 4B ≈ 76.6 MB"
echo "  压缩率 = 1 - (两个 .bin 之和 / 76600000), 预期 ≈ 99.46%"
