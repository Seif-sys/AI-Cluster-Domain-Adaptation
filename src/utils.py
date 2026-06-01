"""
utils.py
========

Shared helper functions used by all scripts.

Keep this file boring and reliable:
    - seed setting
    - device selection
    - saving/loading checkpoints
    - JSON helpers
    - average tracker
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Fix randomness for reproducible experiments.

    This does not guarantee 100% identical results on every machine,
    but it makes the experiment much more repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More deterministic GPU behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> None:
    """Create a folder if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


class AverageMeter:
    """
    Tracks a running average.

    Useful for average loss over many batches.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a Python dictionary as a readable JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_acc: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model weights to disk.

    A checkpoint is useful because:
        1. Deep CORAL can start from the best source-only model.
        2. You can keep the best epoch, not just the last epoch.
        3. If a job crashes, you still have saved progress.
    """
    path = Path(path)
    ensure_dir(path.parent)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "extra": extra or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
