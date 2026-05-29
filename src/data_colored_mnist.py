import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

DIGIT_COLORS = {
    0: (220,  50,  50),   # red
    1: ( 50, 180,  50),   # green
    2: ( 50,  50, 220),   # blue
    3: (220, 180,  50),   # yellow
    4: (180,  50, 220),   # purple
    5: ( 50, 200, 200),   # cyan
    6: (220, 120,  50),   # orange
    7: (140,  80,  40),   # brown
    8: (200,  50, 150),   # pink
    9: ( 80, 160,  80),   # olive-green
}


def _colorize(gray_img_np: np.ndarray, color: tuple) -> np.ndarray:
    """Apply a solid color to the foreground (bright pixels) of a grayscale digit image"""
    rgb = np.zeros((*gray_img_np.shape, 3), dtype=np.uint8)
    mask = gray_img_np > 127          # foreground pixels
    for c, val in enumerate(color):
        rgb[:, :, c][mask] = val
    return rgb


def _assign_color(label: int, correct_color_prob: float, rng: np.random.Generator) -> tuple:
    if rng.random() < correct_color_prob:
        return DIGIT_COLORS[label]
    other = rng.integers(0, 9)       
    if other >= label:
        other += 1
    return DIGIT_COLORS[int(other)]


class ColoredMNIST(Dataset):
    def __init__(
        self,
        mnist_data,
        color_prob: float = 0.99,
        seed: int = 42,
        transform=None,
        binary_labels: bool = False,
        domain_label: int = 0,
    ):
        self.color_prob   = color_prob
        self.transform    = transform
        self.binary_labels = binary_labels
        self.domain_label  = domain_label
 
        rng = np.random.default_rng(seed)
 
        self.images = []
        self.labels = []
 
        for img_tensor, label in mnist_data:
            label = int(label)
            gray  = np.array(img_tensor)          # (H, W) uint8
 
            color    = _assign_color(label, color_prob, rng)
            rgb      = _colorize(gray, color)     # (H, W, 3) uint8
 
            self.images.append(rgb)
            self.labels.append(0 if (binary_labels and label < 5) else
                               1 if (binary_labels and label >= 5) else label)
 
    def __len__(self) -> int:
        return len(self.labels)
 
    def __getitem__(self, idx: int):
        rgb   = self.images[idx]                  
        label = self.labels[idx]
 
        pil_img = Image.fromarray(rgb)
 
        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)
 
        return img_tensor, torch.tensor(label, dtype=torch.long), self.domain_label


# FIX: Make these standalone functions (remove indentation)
def _default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),          # give CNN a bit more spatial room
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])


def get_colored_mnist_loaders(
    root: str = "./data",
    batch_size: int = 64,
    source_color_prob: float = 0.99,
    target_color_prob: float = 0.10,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    binary_labels: bool = False,
    transform=None,
):
    
    if transform is None:
        transform = _default_transform()

    # Download raw MNIST once
    raw_train_plain = datasets.MNIST(root, train=True, download=True,
                                    transform=transforms.Grayscale())
    raw_test_plain  = datasets.MNIST(root, train=False, download=True,
                                    transform=transforms.Grayscale())

    src_full  = ColoredMNIST(raw_train_plain, color_prob=source_color_prob,
                            seed=seed,        transform=transform,
                            binary_labels=binary_labels, domain_label=0)
    src_test  = ColoredMNIST(raw_test_plain,  color_prob=source_color_prob,
                            seed=seed + 100,  transform=transform,
                            binary_labels=binary_labels, domain_label=0)

    # Train / val split on source
    n_val   = int(len(src_full) * val_fraction)
    n_train = len(src_full) - n_val
    src_train, src_val = random_split(
        src_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    tgt_train = ColoredMNIST(raw_train_plain, color_prob=target_color_prob,
                            seed=seed + 200,  transform=transform,
                            binary_labels=binary_labels, domain_label=1)
    tgt_test  = ColoredMNIST(raw_test_plain,  color_prob=target_color_prob,
                            seed=seed + 300,  transform=transform,
                            binary_labels=binary_labels, domain_label=1)

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)

    return (
        _loader(src_train, shuffle=True),    # source_train_loader
        _loader(src_test,  shuffle=False),   # source_test_loader
        _loader(tgt_train, shuffle=True),    # target_train_loader
        _loader(tgt_test,  shuffle=False),   # target_test_loader
        _loader(src_val,   shuffle=False),   # val_loader
    )


if __name__ == "__main__":
    src_train, src_test, tgt_train, tgt_test, val = get_colored_mnist_loaders(
        root="./data", batch_size=64, seed=42
    )

    imgs, labels, domains = next(iter(src_train))
    print(f"Source train batch | images: {imgs.shape}  labels: {labels.shape}  domain: {domains[0].item()}")

    imgs, labels, domains = next(iter(tgt_train))
    print(f"Target train batch | images: {imgs.shape}  labels: {labels.shape}  domain: {domains[0].item()}")

    print("Smoke-test passed.")