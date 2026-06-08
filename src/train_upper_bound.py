"""
train_upper_bound.py
====================

Train target-supervised upper bound.

This uses target labels. That is allowed ONLY because this is the upper bound.
It is NOT unsupervised domain adaptation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_colored_mnist import get_colored_mnist_loaders
from evaluate import evaluate, print_report, save_confusion_matrix_plot
from models import get_model
from utils import AverageMeter, get_device, load_checkpoint, save_checkpoint, save_json, set_seed


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train target-supervised upper bound")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--source_checkpoint", type=str, default="checkpoints/source_only_best.pt")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/upper_bound")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/upper_bound_best.pt")
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return loss_meter.average, correct / total


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)
    print("WARNING: This is supervised upper bound. Target labels ARE used.")

    loaders = get_colored_mnist_loaders(
        batch_size=args.batch_size,
        source_correlation=args.source_correlation,
        target_correlation=args.target_correlation,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_samples=args.max_train_samples,
    )

    model = get_model(
        num_classes=2,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Fine-tuning from source checkpoint is a good upper-bound comparison.
    if Path(args.source_checkpoint).exists():
        load_checkpoint(model, args.source_checkpoint, device=device)
        print("Loaded source checkpoint:", args.source_checkpoint)
    else:
        print("WARNING: source checkpoint not found. Training upper bound from scratch.")

    best_val_acc = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["target_train"], criterion, optimizer, device
        )

        # For the upper bound, target validation labels are allowed.
        val_result = evaluate(model, loaders["target_val"], criterion, device, num_classes=2)
        val_acc = val_result["accuracy"]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                path=args.checkpoint_path,
                epoch=epoch,
                val_acc=val_acc,
                extra=vars(args),
            )

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "target_val_acc": val_acc})

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"target train acc: {train_acc:.4f} | "
            f"target val acc: {val_acc:.4f}"
        )

    training_time = time.time() - start_time

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    source_test = evaluate(model, loaders["source_test"], criterion, device, num_classes=2)
    target_test = evaluate(model, loaders["target_test"], criterion, device, num_classes=2)

    print_report("Target test upper bound", target_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_plot(
        target_test["confusion_matrix"],
        output_dir / "target_confusion_matrix.png",
        class_names=["0-4", "5-9"],
    )

    save_json(
        {
            "method": "target_supervised_upper_bound",
            "note": "Target labels are used here. This is NOT UDA.",
            "args": vars(args),
            "best_val_acc": best_val_acc,
            "training_time_seconds": training_time,
            "history": history,
            "source_test": source_test,
            "target_test": target_test,
        },
        output_dir / "results.json",
    )

    print("Saved upper-bound results to", output_dir)


if __name__ == "__main__":
    main()
