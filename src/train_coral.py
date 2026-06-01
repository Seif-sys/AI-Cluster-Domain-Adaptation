"""
train_coral.py
==============

Train Deep CORAL adaptation.

Deep CORAL uses:
    source images + source labels
    target images WITHOUT target labels

Loss:
    total = classification_loss_on_source + lambda_coral * coral_loss
"""

from __future__ import annotations

import argparse
import time
from itertools import cycle
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_colored_mnist import get_colored_mnist_loaders
from evaluate import evaluate, negative_transfer_check, print_report, save_confusion_matrix_plot
from models import get_model
from utils import AverageMeter, get_device, load_checkpoint, save_checkpoint, save_json, set_seed


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Deep CORAL adaptation")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--lambda_coral", type=float, default=1.0)
    parser.add_argument("--source_checkpoint", type=str, default="checkpoints/source_only_best.pt")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="results/coral")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/coral_best.pt")

    return parser.parse_args()


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    Compute CORAL loss between source and target features.

    It compares covariance matrices. In simple words:
        Are source features and target features spread in a similar way?
    """
    feature_dim = source_features.size(1)

    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    source_cov = (source_centered.T @ source_centered) / (source_features.size(0) - 1)
    target_cov = (target_centered.T @ target_centered) / (target_features.size(0) - 1)

    return torch.mean((source_cov - target_cov) ** 2) / (4 * feature_dim * feature_dim)


def train_one_epoch_coral(model, source_loader, target_loader, criterion, optimizer, device, lambda_coral):
    model.train()
    total_meter = AverageMeter()
    ce_meter = AverageMeter()
    coral_meter = AverageMeter()
    correct = 0
    total = 0

    # cycle(target_loader) means: if target loader ends, start again.
    target_iter = cycle(target_loader)

    for source_images, source_labels in source_loader:
        target_images, _target_labels = next(target_iter)
        # _target_labels exists in the dataset, but we intentionally ignore it.

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        source_features = model.get_features(source_images)
        target_features = model.get_features(target_images)
        source_logits = model.classify(source_features)

        classification_loss = criterion(source_logits, source_labels)
        alignment_loss = coral_loss(source_features, target_features)
        total_loss = classification_loss + lambda_coral * alignment_loss

        total_loss.backward()
        optimizer.step()

        n = source_labels.size(0)
        total_meter.update(total_loss.item(), n)
        ce_meter.update(classification_loss.item(), n)
        coral_meter.update(alignment_loss.item(), n)

        predictions = torch.argmax(source_logits, dim=1)
        correct += (predictions == source_labels).sum().item()
        total += n

    return {
        "train_total_loss": total_meter.average,
        "train_ce_loss": ce_meter.average,
        "train_coral_loss": coral_meter.average,
        "train_source_acc": correct / total,
    }


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)
    print("Lambda CORAL:", args.lambda_coral)

    loaders = get_colored_mnist_loaders(
        batch_size=args.batch_size,
        source_correlation=args.source_correlation,
        target_correlation=args.target_correlation,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_samples=args.max_train_samples,
    )

    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Deep CORAL should start from the source-only model.
    if Path(args.source_checkpoint).exists():
        load_checkpoint(model, args.source_checkpoint, device=device)
        print("Loaded source checkpoint:", args.source_checkpoint)
    else:
        print("WARNING: source checkpoint not found. Training CORAL from scratch.")

    # Accuracy before adaptation, using the source checkpoint.
    target_before = evaluate(model, loaders["target_test"], criterion, device, num_classes=2)

    best_val_acc = -1.0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch_coral(
            model=model,
            source_loader=loaders["source_train"],
            target_loader=loaders["target_train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_coral=args.lambda_coral,
        )

        # Checkpoint selection uses source validation only.
        val_result = evaluate(model, loaders["source_val"], criterion, device, num_classes=2)
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

        epoch_result = {"epoch": epoch, **train_result, "source_val_acc": val_acc}
        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"src train acc: {train_result['train_source_acc']:.4f} | "
            f"source val acc: {val_acc:.4f} | "
            f"coral loss: {train_result['train_coral_loss']:.8f}"
        )

    training_time = time.time() - start_time

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    source_test = evaluate(model, loaders["source_test"], criterion, device, num_classes=2)
    target_after = evaluate(model, loaders["target_test"], criterion, device, num_classes=2)

    transfer = negative_transfer_check(
        source_only_target_acc=target_before["accuracy"],
        adapted_target_acc=target_after["accuracy"],
    )

    print_report("Target before CORAL", target_before)
    print_report("Target after CORAL", target_after)
    print("Negative transfer:", transfer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_plot(
        target_after["confusion_matrix"],
        output_dir / "target_confusion_matrix.png",
        class_names=["0-4", "5-9"],
    )

    save_json(
        {
            "method": "deep_coral_cnn",
            "args": vars(args),
            "best_val_acc": best_val_acc,
            "training_time_seconds": training_time,
            "history": history,
            "target_before_adaptation": target_before,
            "source_test_after_adaptation": source_test,
            "target_test_after_adaptation": target_after,
            "negative_transfer_check": transfer,
        },
        output_dir / "results.json",
    )

    print("Saved CORAL results to", output_dir)


if __name__ == "__main__":
    main()
