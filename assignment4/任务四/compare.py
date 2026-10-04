"""比较 C 推理输出与 PyTorch 参考 x_ref.txt (纯 Python, 无需 torch)。
用法: python3 compare.py [x_out文件]   默认比较 x_out.txt。
"""
from pathlib import Path


def read(path):
    return [float(v) for line in open(path, encoding="utf-8") for v in line.split()]


def main():
    d = Path(__file__).resolve().parent
    import sys
    a_name = sys.argv[1] if len(sys.argv) > 1 else "x_out.txt"
    a = read(d / a_name)
    b = read(d / "x_ref.txt")
    assert len(a) == len(b) == 784, f"维度不符: {a_name}={len(a)}, x_ref={len(b)}"

    errs = [abs(x - y) for x, y in zip(a, b)]
    m = max(errs)
    idx = errs.index(m)
    rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5

    print(f"维度: {len(a)}")
    print(f"最大绝对误差: {m:.6e}  (下标 {idx}, C={a[idx]:.9g}, PyTorch={b[idx]:.9g})")
    print(f"均方根误差(RMSE): {rmse:.6e}")

    if m < 1e-3:
        print("结论: 通过 —— 误差在 float32 舍入量级, C 推理与 PyTorch 一致")
    else:
        print("结论: 异常 —— 误差偏大, 请检查推理逻辑 / 层顺序 / reverse 标志")


if __name__ == "__main__":
    main()
