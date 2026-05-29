"""
utils.py
========
Shared utilities for all training scripts.

Public API
----------
set_seed(seed)                          — fix all random seeds
save_checkpoint(model, optimizer, epoch, val_acc, path)
load_checkpoint(model, path, device)    — load weights into existing model
AverageMeter                            — running mean tracker
get_logger(log_file)                    — file + stdout logger
"""

import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """
    Fix random seeds for Python, NumPy, and PyTorch (CPU + CUDA).
    Also sets CuDNN to deterministic mode.

    Call once at the very top of every training script before any
    data loading or model instantiation.

    Parameters
    ----------
    seed : int  — reported in every results_summary.txt
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # multi-GPU safety
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str,
):
    """
    Save model + optimizer state to disk.

    Saved keys
    ----------
    model_state_dict     : model weights
    optimizer_state_dict : optimizer state (momentum buffers, etc.)
    epoch                : epoch at which checkpoint was saved
    val_acc              : validation accuracy at save time

    Parameters
    ----------
    model     : nn.Module
    optimizer : torch.optim.Optimizer
    epoch     : int
    val_acc   : float
    path      : str  — full file path including filename, e.g. ./outputs/source/best_model.pt
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                epoch,
            "val_acc":              val_acc,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    strict: bool = True,
) -> dict:
    """
    Load model weights from a checkpoint file into an existing model instance.

    Only the model weights are restored; optimizer state and epoch are
    returned as metadata but not applied (each script rebuilds its own
    optimizer from scratch).

    Parameters
    ----------
    model   : nn.Module  — must have the same architecture as the saved model
    path    : str        — path to the .pt checkpoint file
    device  : torch.device
    strict  : bool       — passed to load_state_dict; set False for partial loading

    Returns
    -------
    dict with keys: epoch, val_acc  (for logging)

    Raises
    ------
    FileNotFoundError if path does not exist
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    return {
        "epoch":   checkpoint.get("epoch",   "unknown"),
        "val_acc": checkpoint.get("val_acc",  "unknown"),
    }


# ---------------------------------------------------------------------------
# Running average tracker
# ---------------------------------------------------------------------------

class AverageMeter:
    """
    Track a running mean of a scalar (e.g. loss or accuracy) across batches.

    Usage
    -----
    meter = AverageMeter()
    for batch in loader:
        loss = criterion(...)
        meter.update(loss.item(), batch_size)
    print(meter.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Parameters
        ----------
        val : scalar value for this update (e.g. loss.item())
        n   : number of samples this value represents (usually batch size)
        """
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.4f}, count={self.count})"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(log_file: str, name: str = "da") -> logging.Logger:
    """
    Create a logger that writes to both stdout and a log file simultaneously.

    All training scripts call this once and pass the returned logger through
    to helper functions so every message appears in both places.

    Parameters
    ----------
    log_file : str  — path to .log file (created if it doesn't exist)
    name     : str  — logger name (use different names to avoid duplicate handlers
                      if multiple loggers are created in the same process)

    Returns
    -------
    logging.Logger
    """
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # set_seed
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    assert torch.allclose(a, b), "set_seed failed"
    print("set_seed         ✓")

    # AverageMeter
    meter = AverageMeter()
    for v in [1.0, 2.0, 3.0]:
        meter.update(v, n=1)
    assert abs(meter.avg - 2.0) < 1e-6, f"AverageMeter avg wrong: {meter.avg}"
    meter2 = AverageMeter()
    meter2.update(10.0, n=2)   # represents 2 samples with value 10
    meter2.update(20.0, n=2)   # represents 2 samples with value 20
    assert abs(meter2.avg - 15.0) < 1e-6
    print("AverageMeter     ✓")

    # save / load checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        model = torch.nn.Linear(4, 2)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(model, opt, epoch=5, val_acc=0.91, path=ckpt_path)
        assert os.path.isfile(ckpt_path)

        model2 = torch.nn.Linear(4, 2)
        meta   = load_checkpoint(model2, ckpt_path, device=torch.device("cpu"))
        assert meta["epoch"]   == 5
        assert meta["val_acc"] == 0.91
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    print("save/load ckpt   ✓")

    # logger
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "test.log")
        logger   = get_logger(log_path, name="test_logger")
        logger.info("Logger smoke-test line.")
        assert os.path.isfile(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "Logger smoke-test line." in content
    print("get_logger       ✓")

    print("\nAll smoke-tests passed.")