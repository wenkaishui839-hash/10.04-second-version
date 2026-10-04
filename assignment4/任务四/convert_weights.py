#!/usr/bin/env python3
"""convert_weights.py — 把任务三导出的文本权重转成二进制 (.bin), 供任务四 fread 读取。

用法:
    python3 convert_weights.py [circulant_weights.txt] [scaling_weights.txt]

两个参数缺省为当前目录下的同名文件; 输出 .bin 与输入 .txt 同目录同名。
二进制格式见 load_weights.c 顶部注释 (little-endian, 与 RISC-V / x86 主机一致)。

转换用 float32 (IEEE754 单精度) 存值, 与原先 C 端 strtod -> (float) 的结果
逐位一致, 因此推理数值结果与转换前完全相同。末尾打印的"校验和"与
load_weights_test (main.c) 的 weight_sum 算法一致, 两者应逐位相等。
"""
import os
import struct
import sys


def f32(v):
    """double -> float32 (round-to-nearest), 以 Python double 精确表示结果。"""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def parse_keyval(text):
    """把 key=value 文本解析成 {key: value} (value 为去掉首尾空白的字符串)。"""
    out = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def convert_circulant(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        text = f.read()

    layers = []
    cur = None
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("["):
            if cur is not None:
                layers.append(cur)
            cur = {"name": s.strip("[]")}
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key, val = key.strip(), val.strip()
        if key == "in_features":
            cur["in_features"] = int(val)
        elif key == "out_features":
            cur["out_features"] = int(val)
        elif key == "block_size":
            cur["block_size"] = int(val)
        elif key == "n_block_rows":
            cur["n_block_rows"] = int(val)
        elif key == "n_block_cols":
            cur["n_block_cols"] = int(val)
        elif key == "weight":
            cur["weight"] = [f32(float(x)) for x in val.split()]
        elif key == "bias":
            cur["bias"] = [f32(float(x)) for x in val.split()]
    if cur is not None:
        layers.append(cur)

    checksum = 0.0
    total = 0
    with open(path_out, "wb") as f:
        f.write(struct.pack("<II", 0x43495243, len(layers)))  # "CIRC", n_layers
        for L in layers:
            nr, nc, bs = L["n_block_rows"], L["n_block_cols"], L["block_size"]
            w, b = L["weight"], L["bias"]
            f.write(struct.pack("<iiiii", L["in_features"], L["out_features"],
                                bs, nr, nc))
            f.write(struct.pack("<%df" % len(w), *w))
            f.write(struct.pack("<%df" % len(b), *b))
            checksum += sum(w) + sum(b)   # 与 main.c 的累加顺序一致: 先 weight 再 bias
            total += len(w) + len(b)

    print("[convert] %s -> %s" % (path_in, path_out))
    print("          层数 %d, 循环层参数量 %d" % (len(layers), total))
    return checksum, total


def convert_scaling(path_in, path_out):
    d = parse_keyval(open(path_in, "r", encoding="utf-8").read())
    n = int(d["num_features"])
    log_scale = [f32(float(x)) for x in d["log_scale"].split()]
    scale = [f32(float(x)) for x in d["scale"].split()]

    with open(path_out, "wb") as f:
        f.write(struct.pack("<Ii", 0x5343414C, n))  # "SCAL", num_features
        f.write(struct.pack("<%df" % n, *log_scale))
        f.write(struct.pack("<%df" % n, *scale))

    checksum = sum(log_scale)   # main.c 只累加 log_scale (scale 是派生值)
    print("[convert] %s -> %s" % (path_in, path_out))
    print("          特征数 %d" % n)
    return checksum


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    circ_in = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "circulant_weights.txt")
    scale_in = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "scaling_weights.txt")
    circ_out = os.path.splitext(circ_in)[0] + ".bin"
    scale_out = os.path.splitext(scale_in)[0] + ".bin"

    c_sum, c_total = convert_circulant(circ_in, circ_out)
    s_sum = convert_scaling(scale_in, scale_out)

    total_params = c_total + 1024   # 缩放层 log_scale 1024 个
    print("\n总参数量 : %d (期望 105472)" % total_params)
    print("校验和   : %.9g  (应与 load_weights_test 的 weight_sum 逐位一致)" % (c_sum + s_sum))


if __name__ == "__main__":
    main()
