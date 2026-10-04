"""生成固定噪声 Z 与 PyTorch 参考输出 (自包含, 只需 torch, 无需 pytorch_generative 包)。

用 torch.fft 逐行复现 PaddedBlockCirculantNICE 的生成方向前向 (Z -> X), 权重从
任务三导出的两个 txt 文件读取。torch.fft.rfft/irfft 正是模型 BlockCirculantLinear
内部使用的算子, 因此本参考与训练模型的数学完全一致, 是 C 推理的权威比对基准。

用法 (在虚拟机 torch 环境, 用 python3):
    python3 gen_reference.py [circulant_weights.txt] [scaling_weights.txt]

输出 (写到脚本所在目录):
    z_input.txt   1024 维固定噪声 (torch.manual_seed(0) 的 randn, temp=1.0)
    x_ref.txt     784 维参考输出
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

N_PADDED = 1024
N_ORIGINAL = 784
N_BLOCK = 256


# ---------------- 权重解析 (与 load_weights.c 同格式) ----------------

def parse_circulant(path):
    """解析 circulant_weights.txt 的 24 个 block-circulant 层。"""
    layers, cur = [], None
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            cur = {"name": line.strip("[]")}
            layers.append(cur)
            continue
        k, sep, v = line.partition("=")
        if not cur or not sep:
            continue
        if k in ("weight", "bias"):
            cur[k] = [float(t) for t in v.split()]
        elif k in ("in_features", "out_features", "block_size",
                   "n_block_rows", "n_block_cols"):
            cur[k] = int(v)
        else:
            cur[k] = v
    return layers


def parse_scaling(path):
    """解析 scaling_weights.txt, 返回 scale (1024 维) 与 log_scale。"""
    info = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        k, sep, v = line.partition("=")
        if not sep:
            continue
        if k == "num_features":
            info[k] = int(v)
        elif k in ("log_scale", "scale"):
            info[k] = [float(t) for t in v.split()]
    return info


# ---------------- 复现 BlockCirculantLinear.forward (torch.fft) ----------------

def block_circulant_linear(weight, bias, x):
    """复现 circulant_linear.py 的 BlockCirculantLinear.forward。

    weight: (n_block_rows, n_block_cols, block_size), bias: (out_features,)
    """
    n_block_rows, n_block_cols, block_size = weight.shape
    out_features = bias.shape[0]
    x = x.reshape(-1, n_block_cols, block_size)
    x_fft = torch.fft.rfft(x, n=block_size, dim=-1)
    w_fft = torch.fft.rfft(weight, n=block_size, dim=-1)
    y_fft = (x_fft[:, None, :, :] * w_fft[None, :, :, :]).sum(dim=2)
    y = torch.fft.irfft(y_fft, n=block_size, dim=-1)
    y = y.reshape(-1, n_block_rows * block_size)[..., :out_features]
    return y + bias


def build_tensors(layer):
    rows, cols, bs = layer["n_block_rows"], layer["n_block_cols"], layer["block_size"]
    weight = torch.tensor(layer["weight"], dtype=torch.float32).reshape(rows, cols, bs)
    bias = torch.tensor(layer["bias"], dtype=torch.float32)
    return weight, bias


def coupling_net(layers, block_idx, h):
    """block_idx 块内耦合小网络: 512->1024->(1024->1024 x4)->512 (末层无 ReLU)。"""
    L = layers[block_idx * 6:block_idx * 6 + 6]
    w, b = build_tensors(L[0])
    a = F.relu(block_circulant_linear(w, b, h))      # 层 0: 512 -> 1024
    for i in range(1, 5):                             # 层 1..4: 1024 -> 1024
        w, b = build_tensors(L[i])
        a = F.relu(block_circulant_linear(w, b, a))
    w, b = build_tensors(L[5])
    return block_circulant_linear(w, b, a)            # 层 5: 1024 -> 512


def sample(layers, scale, z):
    """复现 PaddedBlockCirculantNICE.sample(): scaling.inverse -> 逆序 block 3..0 -> 裁剪 784。"""
    half = N_PADDED // 2
    x = z / scale                                   # ScalingLayer.inverse
    reverse_flag = [0, 1, 0, 1]
    for bi in range(3, -1, -1):
        h1, h2 = x[:, :half], x[:, half:]
        net_out = coupling_net(layers, bi, h2 if reverse_flag[bi] else h1)
        if reverse_flag[bi]:
            h1 = h1 - net_out
        else:
            h2 = h2 - net_out
        x = torch.cat((h1, h2), dim=1)
    return x[:, :N_ORIGINAL]


def write_floats(path, tensor):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.9g}" for v in tensor.reshape(-1).tolist()) + "\n")


def main():
    here = Path(__file__).resolve().parent
    default_runs = (here.parent / "任务三(1)" / "任务三" / "pytorch-generative-master"
                    / "runs" / "padded_block_circ_nice_e50")
    circ_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_runs / "circulant_weights.txt"
    scale_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_runs / "scaling_weights.txt"

    layers = parse_circulant(circ_path)
    info = parse_scaling(scale_path)
    scale = torch.tensor(info["scale"], dtype=torch.float32)

    assert len(layers) == 24, f"层数不符: {len(layers)}"
    assert scale.numel() == N_PADDED, f"scaling 维度不符: {scale.numel()}"
    assert all(L["block_size"] == N_BLOCK for L in layers), "block_size 必须为 256"

    torch.manual_seed(0)
    z = torch.randn((1, N_PADDED), dtype=torch.float32)   # temp = 1.0

    with torch.no_grad():
        x = sample(layers, scale, z)

    write_floats(here / "z_input.txt", z)
    write_floats(here / "x_ref.txt", x)

    z4 = z.reshape(-1)[:4].tolist()
    x16 = x.reshape(-1)[:16].tolist()
    print(f"已生成 z_input.txt ({z.numel()} 维) 与 x_ref.txt ({x.numel()} 维)")
    print(f"z 前 4 个 : {['%.6f' % v for v in z4]}")
    print(f"x 前 16 个: {['%.6f' % v for v in x16]}")
    print(f"x 值域   : [{x.min():.4f}, {x.max():.4f}]  (应大致落在 [0,1] 附近)")


if __name__ == "__main__":
    main()
