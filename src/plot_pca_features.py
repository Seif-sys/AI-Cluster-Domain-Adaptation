import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from data_colored_mnist import get_colored_mnist_loaders
from models import SmallCNN
from utils import get_device, set_seed


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load model weights from a checkpoint.

    This function supports different checkpoint formats:
    - {"model_state_dict": ...}
    - {"model": ...}
    - raw state_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)


def extract_features(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Extract internal CNN features.

    This supports two model styles:
    1. model.get_features(images)
    2. model(images, return_features=True)
    """
    if hasattr(model, "get_features"):
        return model.get_features(images)

    output = model(images, return_features=True)

    if isinstance(output, tuple):
        logits, features = output
        return features

    raise RuntimeError(
        "Model does not support feature extraction. "
        "It needs either get_features() or forward(..., return_features=True)."
    )


def collect_domain_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 20,
):
    """
    Collect feature vectors from a DataLoader.

    We do not need the full dataset for PCA.
    A few batches are enough for visualization.
    """
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break

            images = batch[0].to(device)
            labels = batch[1]

            features = extract_features(model, images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels


def plot_pca_for_checkpoint(
    checkpoint_path: str,
    output_path: str,
    title: str,
    target_correlation: float,
    source_correlation: float,
    batch_size: int,
    seed: int,
    max_batches: int,
):
    set_seed(seed)
    device = get_device()

    loaders = get_colored_mnist_loaders(
        batch_size=batch_size,
        source_correlation=source_correlation,
        target_correlation=target_correlation,
        seed=seed,
    )

    model = SmallCNN(num_classes=2).to(device)
    load_model_weights(model, checkpoint_path, device)

    source_features, source_labels = collect_domain_features(
        model,
        loaders["source_test"],
        device,
        max_batches=max_batches,
    )

    target_features, target_labels = collect_domain_features(
        model,
        loaders["target_test"],
        device,
        max_batches=max_batches,
    )

    combined_features = np.concatenate([source_features, target_features], axis=0)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(combined_features)

    source_pca = pca_features[: len(source_features)]
    target_pca = pca_features[len(source_features):]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        source_pca[:, 0],
        source_pca[:, 1],
        alpha=0.45,
        s=12,
        label="Source domain",
    )

    plt.scatter(
        target_pca[:, 0],
        target_pca[:, 1],
        alpha=0.45,
        s=12,
        label="Target domain",
    )

    plt.title(title)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved PCA plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_checkpoint", type=str, required=True)
    parser.add_argument("--coral_checkpoint", type=str, required=True)

    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_batches", type=int, default=20)

    parser.add_argument("--output_dir", type=str, default="results_cloud/final/plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pca_for_checkpoint(
        checkpoint_path=args.source_checkpoint,
        output_path=output_dir / "pca_source_only.png",
        title="PCA Feature Space: Source-only CNN",
        target_correlation=args.target_correlation,
        source_correlation=args.source_correlation,
        batch_size=args.batch_size,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    plot_pca_for_checkpoint(
        checkpoint_path=args.coral_checkpoint,
        output_path=output_dir / "pca_deep_coral.png",
        title="PCA Feature Space: Deep CORAL",
        target_correlation=args.target_correlation,
        source_correlation=args.source_correlation,
        batch_size=args.batch_size,
        seed=args.seed,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()