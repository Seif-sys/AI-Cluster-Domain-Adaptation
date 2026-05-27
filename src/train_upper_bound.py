"""
train_upper_bound.py
====================
Fine-tune the source-trained model on *labeled* target data.

This is the supervised upper bound — it shows the best accuracy achievable
when target labels are available. It is NOT a domain adaptation method.
Its sole purpose is to establish a performance ceiling for comparison.

Protocol
--------
- Load the best source checkpoint produced by train_source.py.
- Fine-tune on labeled target TRAIN data with standard cross-entropy.
- Select the best checkpoint using a held-out target VALIDATION split.
- Report final accuracy on the target TEST set.

⚠  Target labels ARE used here — that is intentional and expected.
   This script must never be described as a UDA method.

Example
-------
python src/train_upper_bound.py \
    --source_ckpt ./outputs/source/best_model.pt \
    --backbone cnn \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 42 \
    --output_dir ./outputs/upper_bound
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

from data_colored_mnist import get_colored_mnist_loaders
from evaluate import evaluate
from models import get_model
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter, get_logger


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Train supervised upper bound on target domain")

    # Checkpoint
    p.add_argument("--source_ckpt", type=str, required=True,
                   help="Path to best_model.pt from train_source.py")

    # Data
    p.add_argument("--data_root",         type=str,   default="./data")
    p.add_argument("--source_color_prob", type=float, default=0.99)
    p.add_argument("--target_color_prob", type=float, default=0.10)
    p.add_argument("--binary_labels",     action="store_true")
    p.add_argument("--tgt_val_fraction",  type=float, default=0.15,
                   help="Fraction of target train set held out for val (checkpoint selection)")

    # Model
    p.add_argument("--backbone",   type=str,  default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--feat_dim",   type=int,  default=256)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder weights; only fine-tune the classifier head")

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4,
                   help="Lower than train_source.py to avoid catastrophic forgetting")
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=2)

    # Output
    p.add_argument("--output_dir",  type=str,   default="./outputs/upper_bound")

    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch (supervised on labeled target data)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, logger):
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    for imgs, labels, _ in loader:
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

    acc = correct / total
    logger.info(f"  train  loss={loss_meter.avg:.4f}  acc={acc:.4f}")
    return loss_meter.avg, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "train_upper_bound.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device : {device}")
    logger.info(f"Args   : {vars(args)}")
    logger.info("NOTE: This is the supervised upper bound. Target labels ARE used.")

    num_classes = 2 if args.binary_labels else 10

    # ---- Data --------------------------------------------------------------
    _, src_test, tgt_train_full, tgt_test, _ = get_colored_mnist_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        source_color_prob=args.source_color_prob,
        target_color_prob=args.target_color_prob,
        seed=args.seed,
        num_workers=args.num_workers,
        binary_labels=args.binary_labels,
    )

    # Split target train into fine-tune train + val
    # (val used only for checkpoint selection — not for tuning method choices)
    tgt_full_ds = tgt_train_full.dataset
    n_total     = len(tgt_full_ds)
    n_val       = int(n_total * args.tgt_val_fraction)
    n_train     = n_total - n_val

    tgt_train_ds, tgt_val_ds = random_split(
        tgt_full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    from torch.utils.data import DataLoader
    tgt_train_loader = DataLoader(tgt_train_ds, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers,
                                  pin_memory=True)
    tgt_val_loader   = DataLoader(tgt_val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True)

    logger.info(f"Target fine-tune train: {n_train}  |  val: {n_val}  |  test: {len(tgt_test.dataset)}")

    # ---- Model: load source checkpoint ------------------------------------
    model = get_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
    ).to(device)

    load_checkpoint(model, args.source_ckpt, device)
    logger.info(f"Loaded source checkpoint: {args.source_ckpt}")

    # Optionally freeze the encoder (only fine-tune classifier head)
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        logger.info("Encoder frozen — fine-tuning classifier head only.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")

    # ---- Optimizer & scheduler --------------------------------------------
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- CSV log ----------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "tgt_test_acc",
    ])
    writer.writeheader()

    # ---- Training loop ----------------------------------------------------
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, tgt_train_loader, optimizer, criterion, device, logger
        )

        val_loss, val_acc, _, _ = evaluate(model, tgt_val_loader, criterion, device)
        logger.info(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            logger.info(f"  -> New best val acc={best_val_acc:.4f}. Checkpoint saved.")

        _, tgt_acc_peek, _, _ = evaluate(model, tgt_test, criterion, device)

        writer.writerow({
            "epoch":       epoch,
            "train_loss":  round(train_loss,    5),
            "train_acc":   round(train_acc,     5),
            "val_loss":    round(val_loss,      5),
            "val_acc":     round(val_acc,       5),
            "tgt_test_acc": round(tgt_acc_peek, 5),
        })
        csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    logger.info(f"\nFine-tuning complete in {elapsed/60:.1f} min")

    # ---- Final evaluation on best checkpoint ------------------------------
    logger.info("\n=== Final Evaluation (best checkpoint) ===")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, src_acc,  _,              _      = evaluate(model, src_test,  criterion, device,
                                                    num_classes=num_classes)
    _, tgt_acc,  tgt_per_class,  tgt_cm = evaluate(model, tgt_test,  criterion, device,
                                                    num_classes=num_classes)

    logger.info(f"Source test accuracy        : {src_acc:.4f}  (sanity — may drop after fine-tune)")
    logger.info(f"Target test accuracy (UB)   : {tgt_acc:.4f}")
    logger.info(f"Per-class target acc        : {[round(a, 3) for a in tgt_per_class]}")

    # ---- Save results summary ---------------------------------------------
    summary_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Supervised Upper Bound ===\n")
        f.write("NOTE: target labels used for fine-tuning. NOT a UDA method.\n\n")
        f.write(f"backbone:            {args.backbone}\n")
        f.write(f"source_ckpt:         {args.source_ckpt}\n")
        f.write(f"freeze_encoder:      {args.freeze_encoder}\n")
        f.write(f"seed:                {args.seed}\n")
        f.write(f"target_color_prob:   {args.target_color_prob}\n")
        f.write(f"epochs:              {args.epochs}\n")
        f.write(f"lr:                  {args.lr}\n")
        f.write(f"batch_size:          {args.batch_size}\n")
        f.write(f"tgt_val_fraction:    {args.tgt_val_fraction}\n")
        f.write(f"training_time_min:   {elapsed/60:.1f}\n")
        f.write(f"device:              {device}\n\n")
        f.write(f"source_test_acc:     {src_acc:.4f}\n")
        f.write(f"target_test_acc_UB:  {tgt_acc:.4f}\n")
        f.write(f"per_class_tgt_acc:   {[round(a,3) for a in tgt_per_class]}\n")

    logger.info(f"\nResults saved to  {summary_path}")
    logger.info(f"Best checkpoint   {best_ckpt_path}")
    logger.info(f"Metrics CSV       {csv_path}")


if __name__ == "__main__":
    main()