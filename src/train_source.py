"""
train_source.py
===============
Train a source-only baseline on Colored-MNIST source domain.

This script:
  1. Trains a classifier on labeled source data only.
  2. Evaluates on source test set and target test set.
  3. Saves the best checkpoint (by source val accuracy).
  4. Logs per-epoch metrics to a CSV for later plotting.

The target domain is never used during training — only for final evaluation.
Target labels are only read at evaluation time, never during training.

Example
-------
python src/train_source.py \
    --backbone cnn \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --seed 42 \
    --results_dir ./results/source \
    --checkpoints_dir ./checkpoints/source
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_colored_mnist import get_colored_mnist_loaders
from evaluate import evaluate
from models import get_model
from utils import set_seed, save_checkpoint, AverageMeter, get_logger


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train source-only baseline")

    # Data
    p.add_argument("--data_root",         type=str,   default="./data")
    p.add_argument("--source_color_prob", type=float, default=0.99,
                   help="Prob. digit gets its canonical color in source domain")
    p.add_argument("--target_color_prob", type=float, default=0.10,
                   help="Prob. digit gets its canonical color in target domain")
    p.add_argument("--binary_labels",     action="store_true", default=True,
                   help="Use 2-class labels (digit < 5 vs >= 5) instead of 10-class (default: True per project spec)")

    # Model
    p.add_argument("--backbone",  type=str, default="cnn",
                   choices=["cnn", "resnet18"])
    p.add_argument("--feat_dim",  type=int, default=256)
    p.add_argument("--pretrained", action="store_true",
                   help="Use ImageNet pretrained weights (ResNet only)")

    # Training
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--num_workers",type=int,   default=2)

    # Output — matches README project structure
    p.add_argument("--results_dir",     type=str, default="./results/source",
                   help="Where to write logs, metrics CSV, and results summary")
    p.add_argument("--checkpoints_dir", type=str, default="./checkpoints/source",
                   help="Where to save model checkpoints")

    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, logger):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for imgs, labels, _ in loader:          # domain label ignored during source training
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))
        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_acc = correct / total
    logger.info(f"  train  loss={loss_meter.avg:.4f}  acc={train_acc:.4f}")
    return loss_meter.avg, train_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.results_dir,     exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.results_dir, "train_source.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    # ---- Data --------------------------------------------------------------
    src_train, src_test, tgt_train, tgt_test, val_loader = get_colored_mnist_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        source_color_prob=args.source_color_prob,
        target_color_prob=args.target_color_prob,
        seed=args.seed,
        num_workers=args.num_workers,
        binary_labels=args.binary_labels,
    )
    num_classes = 2 if args.binary_labels else 10

    # ---- Model -------------------------------------------------------------
    model = get_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
    ).to(device)
    logger.info(f"Model: {args.backbone}  params={sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer & scheduler ---------------------------------------------
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- CSV log -----------------------------------------------------------
    csv_path = os.path.join(args.results_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc",
        "src_test_acc", "tgt_test_acc",
    ])
    writer.writeheader()

    # ---- Training loop -----------------------------------------------------
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.checkpoints_dir, "best_model.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, src_train, optimizer, criterion, device, logger
        )

        # Validation (source val set — used for early stopping / hparam tuning)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        scheduler.step()

        # Save best checkpoint based on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            logger.info(f"  -> New best val acc={best_val_acc:.4f}. Checkpoint saved.")

        # Lightweight per-epoch target peek (informational only — not used for tuning)
        _, tgt_acc_epoch, _, _ = evaluate(model, tgt_test, criterion, device)

        writer.writerow({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "train_acc":   round(train_acc,  5),
            "val_loss":    round(val_loss,   5),
            "val_acc":     round(val_acc,    5),
            "src_test_acc": "N/A",            # full eval done after training
            "tgt_test_acc": round(tgt_acc_epoch, 5),
        })
        csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    logger.info(f"\nTraining complete in {elapsed/60:.1f} min")

    # ---- Final evaluation on best checkpoint -------------------------------
    logger.info("\n=== Final Evaluation (best checkpoint) ===")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, src_acc, src_per_class, src_cm = evaluate(model, src_test, criterion, device,
                                                  num_classes=num_classes)
    _, tgt_acc, tgt_per_class, tgt_cm = evaluate(model, tgt_test, criterion, device,
                                                  num_classes=num_classes)

    logger.info(f"Source test accuracy : {src_acc:.4f}")
    logger.info(f"Target test accuracy : {tgt_acc:.4f}")
    logger.info(f"Per-class target acc : {[round(a, 3) for a in tgt_per_class]}")
    logger.info(f"Gap (src - tgt)      : {src_acc - tgt_acc:.4f}")
    logger.info(f"Seed: {args.seed}")

    # Save final results summary
    summary_path = os.path.join(args.results_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"backbone:            {args.backbone}\n")
        f.write(f"seed:                {args.seed}\n")
        f.write(f"source_color_prob:   {args.source_color_prob}\n")
        f.write(f"target_color_prob:   {args.target_color_prob}\n")
        f.write(f"epochs:              {args.epochs}\n")
        f.write(f"lr:                  {args.lr}\n")
        f.write(f"batch_size:          {args.batch_size}\n")
        f.write(f"training_time_min:   {elapsed/60:.1f}\n")
        f.write(f"device:              {device}\n\n")
        f.write(f"source_test_acc:     {src_acc:.4f}\n")
        f.write(f"target_test_acc:     {tgt_acc:.4f}\n")
        f.write(f"domain_gap:          {src_acc - tgt_acc:.4f}\n")
        f.write(f"per_class_tgt_acc:   {[round(a,3) for a in tgt_per_class]}\n")

    logger.info(f"\nResults saved to   {summary_path}")
    logger.info(f"Best checkpoint    {best_ckpt_path}")
    logger.info(f"Metrics CSV        {csv_path}")


if __name__ == "__main__":
    main()