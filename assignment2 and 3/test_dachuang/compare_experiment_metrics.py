"""Compare two NICE experiment metric files.

Run from the repository root:
    python test_dachuang/compare_experiment_metrics.py \
        runs/circulant_nice_e50/metrics.json \
        runs/block_circulant_nice_e50/metrics.json

Or with explicit names:
    python test_dachuang/compare_experiment_metrics.py \
        --original runs/original_nice_e1/metrics.json \
        --circulant runs/circulant_nice_e1/metrics.json
"""

import argparse
import json
from pathlib import Path


def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def last_loss(metrics):
    history = metrics.get("history", [])
    if not history:
        return None
    return history[-1].get("loss")


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def model_label(metrics, path):
    path = Path(path)
    model_type = metrics.get("model_type", path.parent.name)
    dirname = path.parent.name
    if "block_circulant" in dirname:
        return "Block-Circulant NICE"
    if "circulant" in dirname:
        return "Circulant-NICE"
    if "original" in dirname or model_type == "original":
        return "Original NICE"
    return dirname


def percentage_change(new_value, old_value):
    if old_value in (None, 0) or new_value is None:
        return "N/A"
    return f"{(new_value / old_value - 1):+.2%}"


def resolve_paths(args):
    if args.metrics:
        if len(args.metrics) != 2:
            raise SystemExit("Please provide exactly two metrics.json paths.")
        return args.metrics[0], args.metrics[1]
    if args.original and args.circulant:
        return args.original, args.circulant
    raise SystemExit(
        "Usage: compare_experiment_metrics.py METRICS_A METRICS_B "
        "or --original PATH --circulant PATH"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="*", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--circulant", type=Path)
    args = parser.parse_args()

    left_path, right_path = resolve_paths(args)
    left = load_metrics(left_path)
    right = load_metrics(right_path)

    left_label = model_label(left, left_path)
    right_label = model_label(right, right_path)
    left_params = left.get("param_count")
    right_params = right.get("param_count")
    param_change = percentage_change(right_params, left_params)
    time_change = percentage_change(right.get("total_seconds"), left.get("total_seconds"))
    memory_change = percentage_change(
        right.get("peak_cuda_memory_mb"), left.get("peak_cuda_memory_mb")
    )
    loss_change = percentage_change(last_loss(right), last_loss(left))

    rows = [
        ("model", left_label, right_label, ""),
        ("parameters", left_params, right_params, param_change),
        (
            "total_seconds",
            fmt(left.get("total_seconds"), 2),
            fmt(right.get("total_seconds"), 2),
            time_change,
        ),
        (
            "peak_cuda_memory_mb",
            fmt(left.get("peak_cuda_memory_mb"), 2),
            fmt(right.get("peak_cuda_memory_mb"), 2),
            memory_change,
        ),
        ("final_eval_loss", fmt(last_loss(left), 4), fmt(last_loss(right), 4), loss_change),
        ("sample_image", left.get("sample_image", "N/A"), right.get("sample_image", "N/A"), ""),
        ("checkpoint", left.get("checkpoint", "N/A"), right.get("checkpoint", "N/A"), ""),
    ]

    print(f"| Metric | {left_label} | {right_label} | Right vs Left |")
    print("| --- | ---: | ---: | ---: |")
    for metric, left_value, right_value, change in rows:
        print(f"| {metric} | {left_value} | {right_value} | {change} |")


if __name__ == "__main__":
    main()
