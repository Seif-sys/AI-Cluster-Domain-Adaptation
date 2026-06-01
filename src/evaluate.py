"""
evaluate.py
===========

Evaluation tools for the project.

This file gives us:
    - average loss
    - overall accuracy
    - per-class accuracy
    - confusion matrix
    - negative transfer check
    - optional confusion matrix plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
) -> Dict:
    """
    Evaluate a model on one dataloader.

    Returns a dictionary, so it is easy to save as JSON later.
    """
    model.eval()

    total_loss = 0.0
    total = 0

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        predictions = torch.argmax(logits, dim=1)

        for true_label, pred_label in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion_matrix[true_label, pred_label] += 1

    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()

    # Per-class accuracy = correct for class / all samples of that class.
    class_totals = confusion_matrix.sum(axis=1)
    per_class_accuracy = []
    for class_index in range(num_classes):
        if class_totals[class_index] == 0:
            per_class_accuracy.append(0.0)
        else:
            per_class_accuracy.append(
                float(confusion_matrix[class_index, class_index] / class_totals[class_index])
            )

    return {
        "loss": float(total_loss / total),
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
    }


def negative_transfer_check(source_only_target_acc: float, adapted_target_acc: float) -> Dict:
    """
    Negative transfer = adaptation made target accuracy worse.
    """
    delta = adapted_target_acc - source_only_target_acc
    return {
        "source_only_target_acc": source_only_target_acc,
        "adapted_target_acc": adapted_target_acc,
        "delta": delta,
        "negative_transfer": delta < 0,
    }


def print_report(name: str, result: Dict) -> None:
    """Print a readable evaluation result."""
    print(f"\n{name}")
    print("-" * len(name))
    print(f"loss:     {result['loss']:.4f}")
    print(f"accuracy: {result['accuracy']:.4f} ({result['accuracy'] * 100:.2f}%)")
    print("per-class accuracy:")
    for i, acc in enumerate(result["per_class_accuracy"]):
        print(f"  class {i}: {acc:.4f}")
    print("confusion matrix:")
    for row in result["confusion_matrix"]:
        print(" ", row)


def save_confusion_matrix_plot(
    confusion_matrix: List[List[int]],
    path: str | Path,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Save a confusion matrix image.

    This is useful for the presentation/report.
    """
    matrix = np.array(confusion_matrix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(matrix.shape[0])]

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
