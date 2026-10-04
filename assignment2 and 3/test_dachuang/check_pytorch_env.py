"""Minimal environment check for the Dachuang NICE task.

Run from the repository root:
    python test_dachuang/check_pytorch_env.py
"""

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch_generative import models


def main():
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    model = models.NICE(
        n_features=784,
        n_coupling_blocks=2,
        n_hidden_layers=2,
        n_hidden_features=16,
    ).to(device)

    x = torch.randn(4, 1, 28, 28, device=device)
    preds, log_det_j = model(x)
    log_prob = -(F.softplus(preds) + F.softplus(-preds)).sum(dim=(1, 2, 3))
    loss = -(log_prob + log_det_j).mean()
    loss.backward()

    with torch.no_grad():
        samples = model.sample(n_samples=2)

    print("NICE forward shape:", tuple(preds.shape))
    print("NICE sample shape:", tuple(samples.shape))
    print("Smoke test: OK")


if __name__ == "__main__":
    main()
