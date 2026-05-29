"""
evaluate.py
===========
Shared evaluation utilities used by all three training scripts.

Public API
----------
evaluate(model, loader, criterion, device, num_classes) 
    -> (avg_loss, accuracy, per_class_acc, confusion_matrix)

print_evaluation_report(name, loss, acc, per_class_acc, confusion_matrix)
    -> pretty-prints a full evaluation block to stdout / logger

Usage
-----
from evaluate import evaluate, print_evaluation_report

loss, acc, per_class, cm = evaluate(model, loader, criterion, device, num_classes=10)
print_evaluation_report("Target test", loss, acc, per_class, cm)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
):
    """
    Evaluate model on all batches in loader.

    Parameters
    ----------
    model       : nn.Module  — must implement forward(x) -> logits
    loader      : DataLoader — yields (images, labels, domain_label) or (images, labels)
    criterion   : loss function (e.g. CrossEntropyLoss)
    device      : torch.device
    num_classes : int  number of classes (10 for full MNIST, 2 for binary)

    Returns
    -------
    avg_loss      : float
    accuracy      : float  overall accuracy in [0, 1]
    per_class_acc : list[float]  per-class accuracy, length = num_classes
    conf_matrix   : np.ndarray  shape (num_classes, num_classes)
                    conf_matrix[true, pred] = count
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Confusion matrix accumulator
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in loader:
        # Support loaders that yield 2 or 3 elements
        imgs   = batch[0].to(device)
        labels = batch[1].to(device)
        # batch[2] = domain label, not needed here

        logits = model(imgs)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        trues = labels.cpu().numpy()

        for t, p in zip(trues, preds):
            conf_matrix[t, p] += 1

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # Overall accuracy
    accuracy = conf_matrix.diagonal().sum() / conf_matrix.sum()

    # Per-class accuracy: TP / (all samples of that class)
    class_totals = conf_matrix.sum(axis=1)                     # shape (num_classes,)
    per_class_acc = np.where(
        class_totals > 0,
        conf_matrix.diagonal() / class_totals,
        0.0,
    ).tolist()

    return avg_loss, float(accuracy), per_class_acc, conf_matrix


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_evaluation_report(
    split_name: str,
    avg_loss: float,
    accuracy: float,
    per_class_acc: list,
    conf_matrix: np.ndarray,
    class_names: list = None,
    logger=None,
):
    """
    Print a formatted evaluation report.

    Parameters
    ----------
    split_name    : label for this evaluation (e.g. "Source test", "Target test")
    avg_loss      : float
    accuracy      : float
    per_class_acc : list[float]
    conf_matrix   : np.ndarray (num_classes, num_classes)
    class_names   : optional list of class name strings
    logger        : if provided, use logger.info(); else print()
    """
    num_classes = len(per_class_acc)
    emit = logger.info if logger is not None else print

    emit(f"\n{'='*55}")
    emit(f"  {split_name}")
    emit(f"{'='*55}")
    emit(f"  Loss     : {avg_loss:.4f}")
    emit(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")

    emit(f"\n  Per-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        name = class_names[i] if class_names else str(i)
        bar  = _ascii_bar(acc, width=20)
        emit(f"    class {name:>4} : {acc:.4f}  {bar}")

    emit(f"\n  Confusion matrix (rows=true, cols=pred):")
    _emit_confusion_matrix(conf_matrix, class_names, emit)
    emit(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Negative transfer check
# ---------------------------------------------------------------------------

def check_negative_transfer(
    src_acc_before: float,
    tgt_acc_before: float,
    tgt_acc_after: float,
    logger=None,
) -> bool:
    """
    Warn if adaptation decreased target accuracy (negative transfer).

    Returns True if negative transfer is detected.
    """
    emit = logger.info if logger is not None else print
    delta = tgt_acc_after - tgt_acc_before

    emit(f"\n  Negative transfer check:")
    emit(f"    Target acc before adaptation : {tgt_acc_before:.4f}")
    emit(f"    Target acc after  adaptation : {tgt_acc_after:.4f}")
    emit(f"    Delta                        : {delta:+.4f}")

    if delta < 0:
        emit(f"  ⚠  NEGATIVE TRANSFER DETECTED (delta={delta:+.4f})")
        return True
    else:
        emit(f"  ✓  No negative transfer (delta={delta:+.4f})")
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ascii_bar(value: float, width: int = 20) -> str:
    """Return a simple ASCII progress bar for a value in [0, 1]."""
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _emit_confusion_matrix(matrix: np.ndarray, class_names: list, emit_fn):
    """Print confusion matrix with aligned columns."""
    n = matrix.shape[0]
    names = [str(class_names[i]) if class_names else str(i) for i in range(n)]

    # Header row
    col_w = max(6, max(len(nm) for nm in names) + 1)
    header = " " * (col_w + 2) + "  ".join(f"{nm:>{col_w}}" for nm in names)
    emit_fn(f"    {header}")

    # Separator
    emit_fn("    " + "-" * len(header))

    # Data rows
    for i, row in enumerate(matrix):
        row_str = "  ".join(f"{val:>{col_w}}" for val in row)
        emit_fn(f"    {names[i]:>{col_w}} | {row_str}")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # Dummy model that always predicts class 0
    class DummyModel(nn.Module):
        def __init__(self, n): super().__init__(); self.n = n
        def forward(self, x):
            out = torch.zeros(x.size(0), self.n)
            out[:, 0] = 10.0        # always predicts class 0
            return out

    n_classes = 4
    model   = DummyModel(n_classes)
    device  = torch.device("cpu")
    imgs    = torch.randn(40, 3, 32, 32)
    labels  = torch.randint(0, n_classes, (40,))
    domains = torch.zeros(40, dtype=torch.long)

    ds     = TensorDataset(imgs, labels, domains)
    loader = DataLoader(ds, batch_size=8)
    crit   = nn.CrossEntropyLoss()

    loss, acc, per_class, cm = evaluate(model, loader, crit, device, num_classes=n_classes)
    print_evaluation_report("Smoke-test split", loss, acc, per_class, cm)
    check_negative_transfer(src_acc_before=0.85, tgt_acc_before=0.60, tgt_acc_after=0.55)
    print("Smoke-test passed.")