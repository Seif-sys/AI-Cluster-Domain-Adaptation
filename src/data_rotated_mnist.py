# data_rotated_mnist.py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms



def get_rotated_mnist_loaders(
    root="./data",
    batch_size=64,
    source_angle=0,
    target_angle=15,
    val_fraction=0.1,
    seed=42,
    num_workers=2,
):
    def make_transform(angle):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # CNN expects 3 channels
            transforms.Resize((32, 32)),
            transforms.RandomRotation((angle, angle)),    # fixed angle, not random
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    src_transform = make_transform(source_angle)
    tgt_transform = make_transform(target_angle)

    # torchvision MNIST returns PIL images when transform is applied
    src_full  = datasets.MNIST(root, train=True,  download=True, transform=src_transform)
    src_test  = datasets.MNIST(root, train=False, download=True, transform=src_transform)
    tgt_train = datasets.MNIST(root, train=True,  download=True, transform=tgt_transform)
    tgt_test  = datasets.MNIST(root, train=False, download=True, transform=tgt_transform)

    # Wrap to add domain_label as third return value
    src_full  = DomainDataset(src_full,  domain_label=0)
    src_test  = DomainDataset(src_test,  domain_label=0)
    tgt_train = DomainDataset(tgt_train, domain_label=1)
    tgt_test  = DomainDataset(tgt_test,  domain_label=1)

    n_val   = int(len(src_full) * val_fraction)
    n_train = len(src_full) - n_val
    src_train, src_val = random_split(
        src_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        _loader(src_train, shuffle=True),
        _loader(src_test,  shuffle=False),
        _loader(tgt_train, shuffle=True),
        _loader(tgt_test,  shuffle=False),
        _loader(src_val,   shuffle=False),
    )


class DomainDataset(Dataset):
    """Wraps any (image, label) dataset to also return a domain_label."""
    def __init__(self, dataset, domain_label):
        self.dataset      = dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, self.domain_label