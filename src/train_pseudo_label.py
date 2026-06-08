import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from data_colored_mnist import get_colored_mnist_loaders
from evaluate import evaluate
from models import get_model
from utils import get_device, set_seed


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load model weights from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)


def create_pseudo_label_dataset(
    model: torch.nn.Module,
    target_loader,
    device: torch.device,
    confidence_threshold: float,
):
    """
    Create pseudo-labels for target images.

    Real target labels are ignored.
    We only use the model predictions.

    Returns:
        TensorDataset with:
        - target images
        - pseudo labels
    """
    model.eval()

    selected_images = []
    selected_labels = []
    selected_confidences = []

    with torch.no_grad():
        for batch in target_loader:
            images = batch[0].to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

            confidence, pseudo_labels = torch.max(probabilities, dim=1)

            keep_mask = confidence >= confidence_threshold

            if keep_mask.sum().item() == 0:
                continue

            selected_images.append(images[keep_mask].cpu())
            selected_labels.append(pseudo_labels[keep_mask].cpu())
            selected_confidences.append(confidence[keep_mask].cpu())

    if len(selected_images) == 0:
        raise RuntimeError(
            "No pseudo-labels selected. "
            "Try lowering --confidence_threshold, e.g. 0.90."
        )

    selected_images = torch.cat(selected_images, dim=0)
    selected_labels = torch.cat(selected_labels, dim=0)
    selected_confidences = torch.cat(selected_confidences, dim=0)

    print(f"Selected pseudo-labeled target samples: {len(selected_labels)}")
    print(f"Average pseudo-label confidence: {selected_confidences.mean().item():.4f}")

    return TensorDataset(selected_images, selected_labels)


def train_one_epoch_mixed(
    model,
    source_loader,
    pseudo_loader,
    criterion,
    optimizer,
    device,
    pseudo_weight: float,
):
    """
    Train on source labels and pseudo-target labels.

    source loss:
        real labels from source domain

    pseudo loss:
        predicted labels from target domain
    """
    model.train()

    total_loss = 0.0
    total = 0

    pseudo_iterator = iter(pseudo_loader)

    for source_images, source_labels in source_loader:
        try:
            pseudo_images, pseudo_labels = next(pseudo_iterator)
        except StopIteration:
            pseudo_iterator = iter(pseudo_loader)
            pseudo_images, pseudo_labels = next(pseudo_iterator)

        source_images = source_images.to(device)
        source_labels = source_labels.to(device)

        pseudo_images = pseudo_images.to(device)
        pseudo_labels = pseudo_labels.to(device)

        optimizer.zero_grad()

        source_logits = model(source_images)
        pseudo_logits = model(pseudo_images)

        source_loss = criterion(source_logits, source_labels)
        pseudo_loss = criterion(pseudo_logits, pseudo_labels)

        loss = source_loss + pseudo_weight * pseudo_loss

        loss.backward()
        optimizer.step()

        batch_size = source_labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

    average_loss = total_loss / total

    return average_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_checkpoint", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)

    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    parser.add_argument("--pseudo_weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/pseudo_label")

    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/pseudo_label_best.pt")

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print("Device:", device)

    loaders = get_colored_mnist_loaders(
        batch_size=args.batch_size,
        source_correlation=args.source_correlation,
        target_correlation=args.target_correlation,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
    )

    model = get_model(
        num_classes=2,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    ).to(device)
    load_model_weights(model, args.source_checkpoint, device)

    pseudo_dataset = create_pseudo_label_dataset(
        model=model,
        target_loader=loaders["target_train"],
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_time = time.time()

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_mixed(
            model=model,
            source_loader=loaders["source_train"],
            pseudo_loader=pseudo_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pseudo_weight=args.pseudo_weight,
        )

        source_result = evaluate(
            model,
            loaders["source_test"],
            criterion,
            device,
            num_classes=2,
        )

        target_result = evaluate(
            model,
            loaders["target_test"],
            criterion,
            device,
            num_classes=2,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss: {train_loss:.4f} | "
            f"source acc: {source_result['accuracy']:.4f} | "
            f"target acc: {target_result['accuracy']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "source_test_accuracy": source_result["accuracy"],
                "target_test_accuracy": target_result["accuracy"],
            }
        )

    training_time = time.time() - start_time

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )

    results = {
        "method": "pseudo_labeling",
        "target_labels_used_for_training": False,
        "confidence_threshold": args.confidence_threshold,
        "pseudo_weight": args.pseudo_weight,
        "training_time_seconds": training_time,
        "history": history,
        "source_test": source_result,
        "target_test": target_result,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Saved checkpoint to:", checkpoint_path)
    print("Saved results to:", output_dir / "results.json")


if __name__ == "__main__":
    main()