import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class BinaryColoredMNIST(Dataset):
    """
    Colored-MNIST dataset for binary classification.

    Original MNIST labels:
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    Binary labels:
        0 = digits 0, 1, 2, 3, 4
        1 = digits 5, 6, 7, 8, 9

    Colors:
        binary label 0 -> red
        binary label 1 -> green

    color_correlation:
        0.99 means color matches the label 99% of the time.
        0.50 means color is almost random.
    """

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        color_correlation: float = 0.99,
        seed: int = 42,
        download: bool = True,
    ):
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )

        self.color_correlation = color_correlation

        digit_labels = np.array(self.mnist.targets)
        self.binary_labels = (digit_labels >= 5).astype(np.int64)

        rng = np.random.default_rng(seed)

        # True means: use the correct color for the label.
        use_correct_color = rng.random(len(self.binary_labels)) < color_correlation

        self.color_labels = self.binary_labels.copy()
        self.color_labels[~use_correct_color] = 1 - self.color_labels[~use_correct_color]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        grayscale_image, _ = self.mnist[index]

        binary_label = int(self.binary_labels[index])
        color_label = int(self.color_labels[index])

        colored_image = self._colorize(grayscale_image, color_label)

        return colored_image, binary_label

    def _colorize(self, grayscale_image: torch.Tensor, color_label: int) -> torch.Tensor:
        """
        Convert one grayscale MNIST image into a colored image.

        grayscale_image shape:
            [1, 28, 28]

        colored_image shape:
            [3, 28, 28]

        Channel 0 = red
        Channel 1 = green
        Channel 2 = blue
        """

        colored_image = torch.zeros(3, 28, 28)

        if color_label == 0:
            # Red image
            colored_image[0] = grayscale_image[0]
        else:
            # Green image
            colored_image[1] = grayscale_image[0]

        return colored_image


def get_colored_mnist_loaders(
    batch_size: int = 64,
    source_correlation: float = 0.99,
    target_correlation: float = 0.50,
    seed: int = 42,
):
    """
    Create DataLoaders for source and target domains.

    Source:
        strong color-label correlation

    Target:
        weaker/random color-label correlation
    """

    source_train = BinaryColoredMNIST(
        train=True,
        color_correlation=source_correlation,
        seed=seed,
    )

    source_test = BinaryColoredMNIST(
        train=False,
        color_correlation=source_correlation,
        seed=seed + 1,
    )

    target_train = BinaryColoredMNIST(
        train=True,
        color_correlation=target_correlation,
        seed=seed + 2,
    )

    target_test = BinaryColoredMNIST(
        train=False,
        color_correlation=target_correlation,
        seed=seed + 3,
    )

    source_train_loader = DataLoader(
        source_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    source_test_loader = DataLoader(
        source_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    target_train_loader = DataLoader(
        target_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    target_test_loader = DataLoader(
        target_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return {
        "source_train": source_train_loader,
        "source_test": source_test_loader,
        "target_train": target_train_loader,
        "target_test": target_test_loader,
    }