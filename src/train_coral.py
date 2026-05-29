"""
train_coral.py
==============
Unsupervised Domain Adaptation via Deep CORAL.

CORAL aligns the second-order statistics (covariance matrices) of source and
target feature distributions. No target labels are used during training —
only unlabeled target images are fed through the encoder.

Loss
----
  L = L_CE(source) + lambda_coral * L_CORAL(source_feats, target_feats)

  L_CORAL = (1 / 4d²) * ||Cov_src - Cov_tgt||_F²

  where d = feature dimension, ||·||_F = Frobenius norm.

Protocol
--------
1. Load best source checkpoint from train_source.py.
2. For each batch: sample a source batch (labeled) + target batch (unlabeled).
3. Compute CE loss on source logits + CORAL loss on paired features.
4. Select best checkpoint on source validation accuracy.
5. Report target test accuracy and compare to source-only baseline.

Reference
---------
Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation",
ECCV 2016 Workshop. https://arxiv.org/abs/1607.01719

Example
-------
python src/train_coral.py \
    --source_ckpt ./outputs/source/best_model.pt \
    --backbone cnn \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_coral 1.0 \
    --seed 42 \
    --output_dir ./outputs/coral
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
from evaluate import evaluate, print_evaluation_report, check_negative_transfer
from models import get_model
from utils import set_seed, save_checkpoint, load_checkpoint, AverageMeter, get_logger


# ---------------------------------------------------------------------------
# CORAL loss
# ---------------------------------------------------------------------------

def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    Compute the CORAL loss between source and target feature batches.

    Parameters
    ----------
    source_features : (B_s, d) tensor  — features from source batch
    target_features : (B_t, d) tensor  — features from target batch (NO labels used)

    Returns
    -------
    scalar tensor — CORAL loss value
    """
    d = source_features.size(1)

    # Mean-center each batch independently
    src = source_features - source_features.mean(dim=0, keepdim=True)
    tgt = target_features - target_features.mean(dim=0, keepdim=True)

    # Covariance matrices: (d, d)
    n_s = src.size(0)
    n_t = tgt.size(0)

    cov_src = (src.T @ src) / (n_s - 1)
    cov_tgt = (tgt.T @ tgt) / (n_t - 1)

    # Frobenius norm squared, normalised by 4d²
    loss = torch.norm(cov_src - cov_tgt, p="fro") ** 2 / (4 * d * d)
    return loss


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description="Deep CORAL unsupervised domain adaptation")

    # Checkpoint
    p.add_argument("--source_ckpt", type=str, required=True,
                   help="Path to best_model.pt from train_source.py")

    # Data
    p.add_argument("--data_root",         type=str,   default="./data")
    p.add_argument("--source_color_prob", type=float, default=0.99)
    p.add_argument("--target_color_prob", type=float, default=0.10)
    p.add_argument("--binary_labels",     action="store_true")

    # Model
    p.add_argument("--backbone",   type=str,  default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--feat_dim",   type=int,  default=256)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--dropout",    type=float, default=0.3)

    # CORAL-specific
    p.add_argument("--lambda_coral", type=float, default=1.0,
                   help="Weight of the CORAL loss term. Tune on val, not test.")

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num_workers", type=int,   default=2)

    # Output
    p.add_argument("--output_dir",  type=str,   default="./outputs/coral")

    return p.parse_args()


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, src_loader, tgt_loader,
    optimizer, criterion, lambda_coral,
    device, logger,
):
    """
    Iterate over source batches. For each source batch, draw one target batch
    (cycling through the target loader if it is shorter than source loader).
    Target labels are NEVER passed to the model or loss.
    """
    model.train()

    ce_meter    = AverageMeter()
    coral_meter = AverageMeter()
    loss_meter  = AverageMeter()
    correct = total = 0

    tgt_iter = iter(tgt_loader)

    for src_imgs, src_labels, _ in src_loader:
        src_imgs   = src_imgs.to(device)
        src_labels = src_labels.to(device)

        # Draw a target batch — labels fetched but immediately discarded
        try:
            tgt_imgs, _, _ = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_imgs, _, _ = next(tgt_iter)
        tgt_imgs = tgt_imgs.to(device)

        # Forward passes
        src_feats  = model.get_features(src_imgs)
        tgt_feats  = model.get_features(tgt_imgs)   # no labels involved
        src_logits = model.classify(src_feats)

        # Losses
        loss_ce    = criterion(src_logits, src_labels)
        loss_coral = coral_loss(src_feats, tgt_feats)
        loss       = loss_ce + lambda_coral * loss_coral

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = src_imgs.size(0)
        ce_meter.update(loss_ce.item(),    bs)
        coral_meter.update(loss_coral.item(), bs)
        loss_meter.update(loss.item(),     bs)

        preds    = src_logits.argmax(dim=1)
        correct += (preds == src_labels).sum().item()
        total   += bs

    train_acc = correct / total
    logger.info(
        f"  train  total={loss_meter.avg:.4f}"
        f"  ce={ce_meter.avg:.4f}"
        f"  coral={coral_meter.avg:.4f}"
        f"  src_acc={train_acc:.4f}"
    )
    return loss_meter.avg, ce_meter.avg, coral_meter.avg, train_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "train_coral.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device       : {device}")
    logger.info(f"Args         : {vars(args)}")
    logger.info("UDA method   : Deep CORAL — target labels NOT used during training.")

    num_classes = 2 if args.binary_labels else 10

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

    # ---- Model: start from source checkpoint -------------------------------
    model = get_model(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        feat_dim=args.feat_dim,
        dropout=args.dropout,
    ).to(device)

    load_checkpoint(model, args.source_ckpt, device)
    logger.info(f"Loaded source checkpoint: {args.source_ckpt}")

    # ---- Record source-only baseline on target (before any adaptation) ----
    criterion = nn.CrossEntropyLoss()
    _, tgt_acc_before, tgt_pc_before, _ = evaluate(
        model, tgt_test, criterion, device, num_classes=num_classes
    )
    _, src_acc_before, _, _ = evaluate(
        model, src_test, criterion, device, num_classes=num_classes
    )
    logger.info(f"\nBefore adaptation:")
    logger.info(f"  Source test acc : {src_acc_before:.4f}")
    logger.info(f"  Target test acc : {tgt_acc_before:.4f}  <-- source-only baseline")

    # ---- Optimizer & scheduler ---------------------------------------------
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- CSV log -----------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "total_loss", "ce_loss", "coral_loss",
        "train_src_acc", "val_loss", "val_acc", "tgt_test_acc",
    ])
    writer.writeheader()

    # ---- Training loop -----------------------------------------------------
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}  [lambda_coral={args.lambda_coral}]")

        total_loss, ce_loss, c_loss, train_acc = train_one_epoch(
            model, src_train, tgt_train,
            optimizer, criterion, args.lambda_coral,
            device, logger,
        )

        # Validate on source val set (no target labels involved)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, best_ckpt_path)
            logger.info(f"  -> New best val acc={best_val_acc:.4f}. Checkpoint saved.")

        _, tgt_acc_peek, _, _ = evaluate(model, tgt_test, criterion, device)

        writer.writerow({
            "epoch":        epoch,
            "total_loss":   round(total_loss,  5),
            "ce_loss":      round(ce_loss,      5),
            "coral_loss":   round(c_loss,       5),
            "train_src_acc":round(train_acc,    5),
            "val_loss":     round(val_loss,     5),
            "val_acc":      round(val_acc,      5),
            "tgt_test_acc": round(tgt_acc_peek, 5),
        })
        csv_file.flush()

    csv_file.close()
    elapsed = time.time() - t0
    logger.info(f"\nAdaptation complete in {elapsed/60:.1f} min")

    # ---- Final evaluation on best checkpoint -------------------------------
    logger.info("\n=== Final Evaluation (best checkpoint) ===")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, src_acc_after,  _,             _      = evaluate(
        model, src_test, criterion, device, num_classes=num_classes)
    _, tgt_acc_after,  tgt_per_class,  tgt_cm = evaluate(
        model, tgt_test, criterion, device, num_classes=num_classes)

    print_evaluation_report(
        "Target test (after CORAL)",
        0.0, tgt_acc_after, tgt_per_class, tgt_cm, logger=logger,
    )

    check_negative_transfer(
        src_acc_before=src_acc_before,
        tgt_acc_before=tgt_acc_before,
        tgt_acc_after=tgt_acc_after,
        logger=logger,
    )

    # ---- Save results summary ---------------------------------------------
    summary_path = os.path.join(args.output_dir, "results_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Deep CORAL — Unsupervised Domain Adaptation ===\n")
        f.write("Target labels were NOT used during adaptation.\n\n")
        f.write(f"backbone:            {args.backbone}\n")
        f.write(f"source_ckpt:         {args.source_ckpt}\n")
        f.write(f"lambda_coral:        {args.lambda_coral}\n")
        f.write(f"seed:                {args.seed}\n")
        f.write(f"source_color_prob:   {args.source_color_prob}\n")
        f.write(f"target_color_prob:   {args.target_color_prob}\n")
        f.write(f"epochs:              {args.epochs}\n")
        f.write(f"lr:                  {args.lr}\n")
        f.write(f"batch_size:          {args.batch_size}\n")
        f.write(f"training_time_min:   {elapsed/60:.1f}\n")
        f.write(f"device:              {device}\n\n")
        f.write(f"--- Results ---\n")
        f.write(f"source_acc_before:   {src_acc_before:.4f}\n")
        f.write(f"target_acc_before:   {tgt_acc_before:.4f}  (source-only baseline)\n")
        f.write(f"source_acc_after:    {src_acc_after:.4f}\n")
        f.write(f"target_acc_after:    {tgt_acc_after:.4f}  (CORAL adapted)\n")
        f.write(f"delta_target:        {tgt_acc_after - tgt_acc_before:+.4f}\n")
        f.write(f"negative_transfer:   {tgt_acc_after < tgt_acc_before}\n")
        f.write(f"per_class_tgt_acc:   {[round(a,3) for a in tgt_per_class]}\n")

    logger.info(f"\nResults saved to  {summary_path}")
    logger.info(f"Best checkpoint   {best_ckpt_path}")
    logger.info(f"Metrics CSV       {csv_path}")


if __name__ == "__main__":
    main()