from __future__ import annotations

import numpy as np
import torch
import torchvision

from utils import get_device


def main() -> None:
    device = get_device()

    print("Setup check")
    print("-----------")
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("NumPy version:", np.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    print("Selected device:", device)

    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print("Tensor device:", x.device)
    print("Tensor:", x)

    print("Setup works.")


if __name__ == "__main__":
    main()