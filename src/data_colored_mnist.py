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


def _colorize(gray_img_np: np.ndarray, color: tuple) -> np.ndarray:     #Apply a solid color to the foreground (bright pixels) of a grayscale digit image
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
 
    # ------------------------------------------------------------------
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
 