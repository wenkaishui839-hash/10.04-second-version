#!/usr/bin/env python3
""" 
任务一：C 语言 Circulant 算子 与 Python 官方 torch.fft 的误差比对脚本

原理：三方比对
  1. C 程序       : y_c   = irdFFT(rdFFT(w) .* rdFFT(x))     (irdfft-main.c)
  2. torch.fft   : y_t   = torch.fft.irfft(torch.fft.rfft(w) * torch.fft.rfft(x), n=N)
  3. numpy 直接卷积: y_d  = O(N^2) 循环卷积真值

用法（Linux 虚拟机，需已安装 numpy + torch）:
  # 1. 先编译 C 程序
  gcc -O2 -o irdfft_test irdfft-main.c -lm

  # 2. 运行比对（默认测试 N = 8, 16, 64, 256, 1024）
  python3 compare_torch.py

  # 自定义二进制路径与规模：
  python3 compare_torch.py --bin ./irdfft_test --sizes 8 16 64 256 1024
"""

import argparse
import os
import shlex
import subprocess
import sys

import numpy as np
import torch

TMP_W = "w.bin"
TMP_X = "x.bin"
TMP_Y = "y.bin"


def c_circulant(binary: str, w: np.ndarray, x: np.ndarray, emu: str = None) -> np.ndarray:
    """调用 C 程序: ./binary <N> <w.bin> <x.bin> <y.bin>，返回 y (float32)。

    emu: 模拟器前缀 (RISC-V 二进制在 x86 上直接执行会 SIGILL, 必须经模拟器):
       --emu "qemu-riscv64 -cpu rv64,v=true,vlen=256"             (推荐, 快)
       --emu "spike --isa=rv64gcv $HOME/riscv/riscv64-unknown-linux-gnu/bin/pk"  (慢但权威)
    """
    n = len(w)
    w.astype(np.float32).tofile(TMP_W)
    x.astype(np.float32).tofile(TMP_X)
    cmd = [binary, str(n), TMP_W, TMP_X, TMP_Y]
    if emu:
        cmd = shlex.split(emu) + cmd
    subprocess.run(cmd, check=True)
    y = np.fromfile(TMP_Y, dtype=np.float32)
    if len(y) != n:
        raise RuntimeError(f"C 程序输出长度 {len(y)} != 期望 {n}，请检查二进制路径")
    return y


def torch_circulant(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """torch.fft 官方实现: y = irfft(rfft(w) * rfft(x), n=N)。"""
    wt = torch.from_numpy(w.astype(np.float32))
    xt = torch.from_numpy(x.astype(np.float32))
    y = torch.fft.irfft(torch.fft.rfft(wt) * torch.fft.rfft(xt), n=len(w))
    return y.numpy()


def direct_circulant(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """O(N^2) 循环卷积真值: y[n] = sum_k w[k] * x[(n-k) mod N]。
    对应循环矩阵 C 的第一列 = w（与 C 程序、torch 约定一致）。"""
    n = len(w)
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    return (w[idx] @ x).astype(np.float32)


def stats(ref: np.ndarray, got: np.ndarray):
    """max abs err / RMSE / 相对误差（按 ref 峰值归一化）。"""
    ref64 = ref.astype(np.float64)
    got64 = got.astype(np.float64)
    diff = np.abs(got64 - ref64)
    max_abs = diff.max()
    rmse = np.sqrt((diff ** 2).mean())
    scale = np.abs(ref64).max()
    rel = max_abs / scale if scale > 0 else float(max_abs)
    return max_abs, rmse, rel


def main():
    ap = argparse.ArgumentParser(description="C Circulant 算子 vs torch.fft 误差比对")
    ap.add_argument("--bin", default="./irdfft_test",
                    help="C 程序路径（默认 ./irdfft_test）")
    ap.add_argument("--emu", default=None,
                    help='模拟器前缀，RVV 二进制必填。例: --emu "qemu-riscv64 -cpu rv64,v=true,vlen=256"')
    ap.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 64, 256, 1024],
                    help="测试的 N 列表，必须是 2 的幂")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    if not os.path.exists(args.bin):
        print(f"[错误] 找不到 C 程序 {args.bin!r}")
        print("       请先编译: gcc -O2 -o irdfft_test irdfft-main.c -lm")
        print("       或用 --bin 指定路径")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    print(f"{'N':>6} | {'C vs torch':>32} | {'C vs 直接卷积':>36} | {'torch vs 直接卷积':>34}")
    print(f"{'':>6} | {'max_abs':>12} {'RMSE':>10} {'rel':>9} | "
          f"{'max_abs':>12} {'RMSE':>10} {'rel':>9} | {'max_abs':>12} {'RMSE':>10}")
    print("-" * 120)

    all_pass = True
    for n in args.sizes:
        if (n & (n - 1)) != 0 or n < 2:
            print(f"[跳过] N={n} 不是 2 的幂")
            continue

        w = rng.standard_normal(n).astype(np.float32)
        x = rng.standard_normal(n).astype(np.float32)

        y_c = c_circulant(args.bin, w, x, args.emu)
        y_t = torch_circulant(w, x)
        y_d = direct_circulant(w, x)

        c_ma, c_rmse, c_rel = stats(y_t, y_c)
        d_ma, d_rmse, d_rel = stats(y_d, y_c)
        t_ma, t_rmse, _ = stats(y_d, y_t)

        # float32 下相对误差约 1e-6 量级，留 10 倍余量
        ok = c_rel < 1e-5
        all_pass &= ok
        flag = "PASS" if ok else "FAIL"
        print(f"{n:>6} | {c_ma:12.3e} {c_rmse:10.3e} {c_rel:9.3e} | "
              f"{d_ma:12.3e} {d_rmse:10.3e} {d_rel:9.3e} | "
              f"{t_ma:12.3e} {t_rmse:10.3e}   [{flag}]")

    print("-" * 120)
    if all_pass:
        print("结论: 全部通过。C 版 Circulant 算子与 torch.fft 结果一致（float32 舍入误差范围内）。")
    else:
        print("结论: 存在 FAIL，请检查 C 实现。")
        sys.exit(1)

    for f in (TMP_W, TMP_X, TMP_Y):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
