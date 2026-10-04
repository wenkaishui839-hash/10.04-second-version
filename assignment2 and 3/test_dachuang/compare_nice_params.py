"""Compare parameter counts of original NICE and Circulant-NICE.

Run from the repository root:
    python test_dachuang/compare_nice_params.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch_generative import models


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    nice = models.NICE(n_features=784)
    circulant_nice = models.CirculantNICE(n_features=784)

    nice_params = count_params(nice)
    circulant_params = count_params(circulant_nice)
    reduction = 1 - circulant_params / nice_params

    print("Original NICE params:", nice_params)
    print("Circulant-NICE params:", circulant_params)
    print(f"Parameter reduction: {reduction:.2%}")


if __name__ == "__main__":
    main()
