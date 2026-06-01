"""
data_colored_mnist.py
=====================

Creates the Colored-MNIST source and target datasets for our project.

Project setup
-------------
Binary classification:
    class 0 = digits 0, 1, 2, 3, 4
    class 1 = digits 5, 6, 7, 8, 9

Source domain:
    color-label correlation is strong, e.g. 0.99.

Target domain:
    color-label correlation is weaker/different, e.g. 0.10 or 0.50.

Important:
    This file creates target labels too, because we need them for evaluation.
    But Deep CORAL training must ignore target labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


class BinaryColoredMNIST(Dataset):
    """
    Custom PyTorch Dataset for binary Colored-MNIST.

    A Dataset must implement:
        __len__      -> tells PyTorch how many samples exist
        __getitem__  -> returns one sample by index

    Returns:
        colored_image: tensor with shape [3, 28, 28]
        binary_label : tensor, 0 or 1
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        color_correlation: float = 0.99,
        seed: int = 42,
        download: bool = True,
    ) -> None:
        # Load original black/white MNIST.
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )

        self.color_correlation = color_correlation

        # Original labels are 0..9.
        digit_labels = np.array(self.mnist.targets)

        # Convert 10 classes into 2 classes:
        # digits 0-4 -> 0, digits 5-9 -> 1.
        self.binary_labels = (digit_labels >= 5).astype(np.int64)

        # Random generator for reproducible color assignment.
        rng = np.random.default_rng(seed)

        # True means: use correct color for the binary label.
        use_correct_color = rng.random(len(self.binary_labels)) < color_correlation

        # color_label controls only the color, not the true training label.
        self.color_labels = self.binary_labels.copy()

        # If correct color is not used, flip color 0 <-> 1.
        self.color_labels[~use_correct_color] = 1 - self.color_labels[~use_correct_color]

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, index: int):
        # self.mnist[index] returns (image, original_digit_label).
        # We ignore the original digit label here because we created binary labels.
        grayscale_image, _ = self.mnist[index]

        binary_label = int(self.binary_labels[index])
        color_label = int(self.color_labels[index])

        colored_image = self._colorize(grayscale_image, color_label)

        return colored_image, torch.tensor(binary_label, dtype=torch.long)

    def _colorize(self, grayscale_image: torch.Tensor, color_label: int) -> torch.Tensor:
        """
        Convert one grayscale image [1, 28, 28] into RGB [3, 28, 28].

        color_label = 0 -> red digit
        color_label = 1 -> green digit
        """
        colored_image = torch.zeros(3, 28, 28)

        if color_label == 0:
            colored_image[0] = grayscale_image[0]  # red channel
        else:
            colored_image[1] = grayscale_image[0]  # green channel

        return colored_image


@dataclass
class LoaderConfig:
    """Small config object to keep DataLoader settings readable."""

    root: str = "data"
    batch_size: int = 64
    source_correlation: float = 0.99
    target_correlation: float = 0.10
    seed: int = 42
    val_fraction: float = 0.10
    num_workers: int = 0
    max_train_samples: Optional[int] = None  # None = use full MNIST train set


def _maybe_limit_dataset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    """
    Optional speed tool.

    Full MNIST train has 60,000 samples. That is already not small.
    If your CPU is slow, you can temporarily set max_train_samples=10000.
    For final results, keep max_train_samples=None.
    """
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _split_train_val(dataset: Dataset, val_fraction: float, seed: int):
    """
    Split a training dataset into train and validation.

    Validation data is used to choose the best checkpoint.
    It is NOT the final test set.
    """
    n_val = int(len(dataset) * val_fraction)
    n_train = len(dataset) - n_val

    return random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def get_colored_mnist_loaders(
    root: str = "data",
    batch_size: int = 64,
    source_correlation: float = 0.99,
    target_correlation: float = 0.10,
    seed: int = 42,
    val_fraction: float = 0.10,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create all loaders needed for the full experiment.

    Returns dictionary with:
        source_train  -> train source-only model
        source_val    -> select best source/checkpoint
        source_test   -> final source evaluation
        target_train  -> Deep CORAL uses images only; upper bound uses labels
        target_val    -> upper bound checkpoint selection
        target_test   -> final target evaluation
    """

    # Use deterministic but different seeds for source/target train/test.
    # Base seed is still 42; offsets make independent color assignments.
    source_full = BinaryColoredMNIST(
        root=root,
        train=True,
        color_correlation=source_correlation,
        seed=seed,
    )
    source_test = BinaryColoredMNIST(
        root=root,
        train=False,
        color_correlation=source_correlation,
        seed=seed + 1,
    )
    target_full = BinaryColoredMNIST(
        root=root,
        train=True,
        color_correlation=target_correlation,
        seed=seed + 2,
    )
    target_test = BinaryColoredMNIST(
        root=root,
        train=False,
        color_correlation=target_correlation,
        seed=seed + 3,
    )

    # Optional speed mode for slow CPU experiments.
    source_full = _maybe_limit_dataset(source_full, max_train_samples, seed=seed + 10)
    target_full = _maybe_limit_dataset(target_full, max_train_samples, seed=seed + 20)

    source_train, source_val = _split_train_val(source_full, val_fraction, seed=seed + 30)
    target_train, target_val = _split_train_val(target_full, val_fraction, seed=seed + 40)

    def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "source_train": make_loader(source_train, shuffle=True),
        "source_val": make_loader(source_val, shuffle=False),
        "source_test": make_loader(source_test, shuffle=False),
        "target_train": make_loader(target_train, shuffle=True),
        "target_val": make_loader(target_val, shuffle=False),
        "target_test": make_loader(target_test, shuffle=False),
    }


if __name__ == "__main__":
    loaders = get_colored_mnist_loaders(batch_size=64)

    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        print(name, images.shape, labels.shape, labels[:8].tolist())

    print("Colored-MNIST loaders work.")
