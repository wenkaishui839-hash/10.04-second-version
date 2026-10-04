"""Run original NICE or Circulant-NICE experiments for the Dachuang task.

Run from the repository root:
    python test_dachuang/run_nice_experiment.py --model-type original --epochs 1
    python test_dachuang/run_nice_experiment.py --model-type circulant --epochs 1
    python test_dachuang/run_nice_experiment.py --model-type partial-block-circ --epochs 1
    python test_dachuang/run_nice_experiment.py --model-type padded-original --epochs 1
    python test_dachuang/run_nice_experiment.py --model-type padded-partial-block-circ --epochs 1
    python test_dachuang/run_nice_experiment.py --model-type padded-block-circ --epochs 1
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

from pytorch_generative import models
from pytorch_generative.nn import BlockCirculantLinear, CirculantLinear


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


def make_model(model_type):
    nice_kwargs = {
        "n_features": 784,
        "n_coupling_blocks": 4,
        "n_hidden_layers": 5,
        "n_hidden_features": 1000,
    }
    if model_type == "original":
        return models.NICE(**nice_kwargs)
    if model_type == "circulant":
        return models.CirculantNICE(**nice_kwargs)
    if model_type == "partial-block-circ":
        return models.PartialBlockCirculantNICE(
            n_features=784,
            n_coupling_blocks=4,
            n_hidden_layers=5,
            n_hidden_features=1024,
            block_size=256,
        )
    if model_type == "padded-original":
        return models.PaddedNICE(
            n_features=784,
            padded_features=1024,
            n_coupling_blocks=4,
            n_hidden_layers=5,
            n_hidden_features=1024,
        )
    if model_type == "padded-partial-block-circ":
        return models.PaddedPartialBlockCirculantNICE(
            n_features=784,
            padded_features=1024,
            n_coupling_blocks=4,
            n_hidden_layers=5,
            n_hidden_features=1024,
            block_size=256,
        )
    if model_type == "padded-block-circ":
        return models.PaddedBlockCirculantNICE(
            n_features=784,
            padded_features=1024,
            n_coupling_blocks=4,
            n_hidden_layers=5,
            n_hidden_features=1024,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def export_circulant_weights(model, output_dir):
    """Export CirculantLinear vectors and biases for C/RISC-V inference."""
    layers = {}
    text_lines = []

    for name, module in model.named_modules():
        if isinstance(module, CirculantLinear):
            layer_record = {
                "type": "circulant",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "circulant_dim": module.circulant_dim,
                "weight": module.weight.detach().cpu(),
                "bias": None if module.bias is None else module.bias.detach().cpu(),
            }

            text_lines.append(f"[{name}]")
            text_lines.append("type=circulant")
            text_lines.append(f"in_features={module.in_features}")
            text_lines.append(f"out_features={module.out_features}")
            text_lines.append(f"circulant_dim={module.circulant_dim}")
            text_lines.append(
                "weight=" + " ".join(f"{v:.9g}" for v in layer_record["weight"].tolist())
            )
        elif isinstance(module, BlockCirculantLinear):
            layer_record = {
                "type": "block_circulant",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "block_size": module.block_size,
                "n_block_rows": module.n_block_rows,
                "n_block_cols": module.n_block_cols,
                "weight": module.weight.detach().cpu(),
                "bias": None if module.bias is None else module.bias.detach().cpu(),
            }

            text_lines.append(f"[{name}]")
            text_lines.append("type=block_circulant")
            text_lines.append(f"in_features={module.in_features}")
            text_lines.append(f"out_features={module.out_features}")
            text_lines.append(f"block_size={module.block_size}")
            text_lines.append(f"n_block_rows={module.n_block_rows}")
            text_lines.append(f"n_block_cols={module.n_block_cols}")
            flat_weight = layer_record["weight"].reshape(-1).tolist()
            text_lines.append("weight=" + " ".join(f"{v:.9g}" for v in flat_weight))
        else:
            continue

        layers[name] = layer_record

        if layer_record["bias"] is not None:
            text_lines.append("bias=" + " ".join(f"{v:.9g}" for v in layer_record["bias"].tolist()))
        else:
            text_lines.append("bias=None")
        text_lines.append("")

    pt_path = output_dir / "circulant_weights.pt"
    txt_path = output_dir / "circulant_weights.txt"
    torch.save(layers, pt_path)
    txt_path.write_text("\n".join(text_lines), encoding="utf-8")
    return pt_path, txt_path, len(layers)


def export_scaling_weights(model, output_dir):
    """Export the NICE scaling layer for deployment."""
    if not hasattr(model, "scaling") or not hasattr(model.scaling, "log_scale"):
        return None, None

    log_scale = model.scaling.log_scale.detach().cpu()
    scale = torch.exp(log_scale)
    record = {
        "type": "scaling",
        "log_scale": log_scale,
        "scale": scale,
    }

    pt_path = output_dir / "scaling_weights.pt"
    txt_path = output_dir / "scaling_weights.txt"
    lines = [
        "[scaling]",
        "type=scaling",
        f"num_features={log_scale.numel()}",
        "log_scale=" + " ".join(f"{v:.9g}" for v in log_scale.reshape(-1).tolist()),
        "scale=" + " ".join(f"{v:.9g}" for v in scale.reshape(-1).tolist()),
    ]
    torch.save(record, pt_path)
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return pt_path, txt_path


def save_outputs(model, optimizer, args, history, total_seconds, peak_cuda_memory_mb):
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=16, temp=args.sample_temp).clamp(0, 1).cpu()

    sample_path = args.logdir / "samples_epoch_last.png"
    checkpoint_path = args.logdir / f"{args.model_type}_nice.pt"
    metrics_path = args.logdir / "metrics.json"
    utils.save_image(samples, sample_path, nrow=4)

    param_count = sum(p.numel() for p in model.parameters())
    checkpoint = {
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "param_count": param_count,
        "history": history,
        "total_seconds": total_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "model_type": args.model_type,
        "device": args.resolved_device,
        "param_count": param_count,
        "total_seconds": total_seconds,
        "peak_cuda_memory_mb": peak_cuda_memory_mb,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "sample_image": str(sample_path),
    }

    scaling_pt_path, scaling_txt_path = export_scaling_weights(model, args.logdir)
    if scaling_pt_path is not None:
        metrics["scaling_weights_pt"] = str(scaling_pt_path)
        metrics["scaling_weights_txt"] = str(scaling_txt_path)

    if args.model_type in (
        "circulant",
        "partial-block-circ",
        "padded-partial-block-circ",
        "padded-block-circ",
    ):
        pt_path, txt_path, n_layers = export_circulant_weights(model, args.logdir)
        metrics["circulant_layer_count"] = n_layers
        metrics["circulant_weights_pt"] = str(pt_path)
        metrics["circulant_weights_txt"] = str(txt_path)

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=[
            "original",
            "circulant",
            "partial-block-circ",
            "padded-original",
            "padded-partial-block-circ",
            "padded-block-circ",
        ],
        default="original",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--logdir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("C:/tmp/data"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--sample-temp", type=float, default=0.7)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    args.resolved_device = device

    if args.logdir is None:
        args.logdir = Path(f"runs/{args.model_type}_nice_e{args.epochs}")
    args.logdir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size)
    model = make_model(args.model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    param_count = sum(p.numel() for p in model.parameters())
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Model type:", args.model_type)
    print("Device:", device)
    print("Parameters:", param_count)
    print("Train batches:", len(train_loader))
    print("Eval batches:", len(test_loader))

    start_time = time.time()
    history = []

    try:
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
                    print(
                        f"epoch {epoch}/{args.epochs} "
                        f"step {step}/{len(train_loader)} loss {avg_loss:.4f}"
                    )

            eval_metrics = evaluate(model, test_loader, device)
            epoch_seconds = time.time() - epoch_start
            history.append({"epoch": epoch, "seconds": epoch_seconds, **eval_metrics})
            print(
                f"epoch {epoch} eval loss {eval_metrics['loss']:.4f} "
                f"time {epoch_seconds:.1f}s"
            )
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")

    total_seconds = time.time() - start_time
    peak_cuda_memory_mb = None
    if device == "cuda":
        peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    metrics = save_outputs(
        model=model,
        optimizer=optimizer,
        args=args,
        history=history,
        total_seconds=total_seconds,
        peak_cuda_memory_mb=peak_cuda_memory_mb,
    )

    print("Done.")
    print("Total seconds:", round(total_seconds, 2))
    print("Peak CUDA memory MB:", None if peak_cuda_memory_mb is None else round(peak_cuda_memory_mb, 2))
    print("Checkpoint:", metrics["checkpoint"])
    print("Samples:", metrics["sample_image"])
    print("Metrics:", args.logdir / "metrics.json")
    if "scaling_weights_pt" in metrics:
        print("Scaling weights .pt:", metrics["scaling_weights_pt"])
        print("Scaling weights .txt:", metrics["scaling_weights_txt"])
    if args.model_type in (
        "circulant",
        "partial-block-circ",
        "padded-partial-block-circ",
        "padded-block-circ",
    ):
        print("Circulant weights .pt:", metrics["circulant_weights_pt"])
        print("Circulant weights .txt:", metrics["circulant_weights_txt"])


if __name__ == "__main__":
    main()
