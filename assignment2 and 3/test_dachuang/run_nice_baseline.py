"""Train the original NICE baseline for the Dachuang task.

This script avoids Windows DataLoader worker issues by using num_workers=0.

Run from the repository root:
    python test_dachuang/run_nice_baseline.py --epochs 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch_generative.models.flow.nice import NICE


def dequantize(x):
    return (x * 255 + torch.rand(x.shape)) / 256


def make_loaders(data_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), dequantize])
    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def nice_loss(preds):
    z, log_det_j = preds
    log_prob = -(F.softplus(z) + F.softplus(-z)).sum(dim=(1, 2, 3))
    objective = log_prob + log_det_j
    return -objective.mean(), log_prob.mean(), log_det_j.mean()


def evaluate(model, loader, device):
    model.eval()
    n_examples = 0
    total_loss = 0.0
    total_log_prob = 0.0
    total_log_det_j = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            loss, log_prob, log_det_j = nice_loss(model(x))
            batch_size = x.shape[0]
            n_examples += batch_size
            total_loss += loss.item() * batch_size
            total_log_prob += log_prob.item() * batch_size
            total_log_det_j += log_det_j.item() * batch_size
    return {
        "loss": total_loss / n_examples,
        "prior_log_likelihood": total_log_prob / n_examples,
        "log_det_J": total_log_det_j / n_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--logdir", type=Path, default=Path("runs/nice_baseline_original"))
    parser.add_argument("--data-dir", type=Path, default=Path("C:/tmp/data"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--sample-temp", type=float, default=0.7)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    args.logdir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size)

    model = NICE(
        n_features=784,
        n_coupling_blocks=4,
        n_hidden_layers=5,
        n_hidden_features=1000,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    param_count = sum(p.numel() for p in model.parameters())
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Baseline: original NICE")
    print("Device:", device)
    print("Parameters:", param_count)
    print("Train batches:", len(train_loader))
    print("Eval batches:", len(test_loader))

    start_time = time.time()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_start = time.time()
        for step, (x, _) in enumerate(train_loader, start=1):
            x = x.to(device)
            optimizer.zero_grad()
            loss, _, _ = nice_loss(model(x))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step == 1 or step % 10 == 0 or step == len(train_loader):
                avg_loss = running_loss / step
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(train_loader)} loss {avg_loss:.4f}")

        eval_metrics = evaluate(model, test_loader, device)
        epoch_seconds = time.time() - epoch_start
        epoch_record = {
            "epoch": epoch,
            "seconds": epoch_seconds,
            **eval_metrics,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch} eval loss {eval_metrics['loss']:.4f} "
            f"time {epoch_seconds:.1f}s"
        )

    total_seconds = time.time() - start_time
    peak_cuda_memory_mb = None
    if device == "cuda":
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=16, temp=args.sample_temp).clamp(0, 1).cpu()
    sample_path = args.logdir / "samples_epoch_last.png"
    utils.save_image(samples, sample_path, nrow=4)

    checkpoint_path = args.logdir / "nice_baseline_original.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "param_count": param_count,
            "history": history,
            "total_seconds": total_seconds,
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
        },
        checkpoint_path,
    )

    metrics = {
        "baseline": "original NICE",
        "device": device,
        "param_count": param_count,
        "total_seconds": total_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "sample_image": str(sample_path),
    }
    metrics_path = args.logdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Done.")
    print("Total seconds:", round(total_seconds, 2))
    print("Peak CUDA memory MB:", None if peak_cuda_memory_mb is None else round(peak_cuda_memory_mb, 2))
    print("Checkpoint:", checkpoint_path)
    print("Samples:", sample_path)
    print("Metrics:", metrics_path)


if __name__ == "__main__":
    main()
