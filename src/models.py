"""
models.py
=========
Feature extractor + classifier head for Colored-MNIST domain adaptation.

Two backbones are provided:
  - SmallCNN   : lightweight 3-conv CNN, good for quick experiments on MNIST-scale data.
  - ResNet18DA : torchvision ResNet-18 with the final FC replaced, supports
                 optional ImageNet pretraining.

Both expose the same interface:
  model.get_features(x)  -> (B, feat_dim) tensor  [used by CORAL loss]
  model.classify(feats)  -> (B, num_classes) logits
  model.forward(x)       -> (B, num_classes) logits  [convenience = get_features + classify]

Usage
-----
from models import SmallCNN, ResNet18DA, get_model

# Quick CNN (no pretrained weights needed)
model = get_model("cnn", num_classes=10)

# ResNet-18 from ImageNet checkpoint (recommended for harder datasets)
model = get_model("resnet18", num_classes=10, pretrained=True)

# Access features directly (needed in train_coral.py)
feats  = model.get_features(images)   # (B, feat_dim)
logits = model.classify(feats)        # (B, num_classes)
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Lightweight CNN backbone
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    """
    3-block convolutional feature extractor followed by a linear classifier.

    Architecture
    ------------
    Input  : (B, 3, 32, 32)   [RGB Colored-MNIST resized to 32x32]
    Block 1: Conv(3->32, 3x3) -> BN -> ReLU -> MaxPool(2)  => (B, 32, 16, 16)
    Block 2: Conv(32->64, 3x3)-> BN -> ReLU -> MaxPool(2)  => (B, 64,  8,  8)
    Block 3: Conv(64->128,3x3)-> BN -> ReLU -> MaxPool(2)  => (B, 128, 4,  4)
    Flatten + Linear(2048->256) -> BN -> ReLU -> Dropout    => (B, 256)  <-- features
    Classifier: Linear(256 -> num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.feat_dim = 256

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 4x4
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.feat_dim),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(self.feat_dim, num_classes)

    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feat_dim) feature vectors — input to the CORAL loss."""
        return self.projector(self.encoder(x))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Map (B, feat_dim) features -> (B, num_classes) logits."""
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: image -> logits."""
        return self.classify(self.get_features(x))


# ---------------------------------------------------------------------------
# ResNet-18 backbone
# ---------------------------------------------------------------------------

class ResNet18DA(nn.Module):
    """
    ResNet-18 with the final FC replaced by a two-stage projection + classifier.

    The split between `get_features` and `classify` is identical to SmallCNN,
    making the two backbones drop-in replacements for each other.

    Architecture (after the standard ResNet conv stack)
    ----------------------------------------------------
    AdaptiveAvgPool -> (B, 512)
    Linear(512 -> feat_dim) -> BN -> ReLU -> Dropout   => (B, feat_dim)  <-- features
    Classifier: Linear(feat_dim -> num_classes)

    Parameters
    ----------
    num_classes : int   number of output classes
    pretrained  : bool  load ImageNet weights (strongly recommended)
    feat_dim    : int   size of the intermediate feature layer (default 256)
    dropout     : float dropout probability in the feature layer
    freeze_bn   : bool  freeze BatchNorm statistics (useful when fine-tuning with
                        small batches in the target domain)
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        feat_dim: int = 256,
        dropout: float = 0.3,
        freeze_bn: bool = False,
    ):
        super().__init__()

        self.feat_dim = feat_dim

        # Load backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Everything up to (but not including) the original FC
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])   # output: (B, 512, 1, 1)

        # Replace FC with a bottleneck projector
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(feat_dim, num_classes)

        if freeze_bn:
            self._freeze_bn()

    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, feat_dim) feature vectors."""
        return self.projector(self.encoder(x))

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Map (B, feat_dim) features -> (B, num_classes) logits."""
        return self.fc(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.get_features(x))

    # ------------------------------------------------------------------
    def _freeze_bn(self):
        """Set all BatchNorm layers to eval mode (frozen running stats)."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_model(
    backbone: str = "cnn",
    num_classes: int = 10,
    pretrained: bool = True,
    feat_dim: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Instantiate and return a model by name.

    Parameters
    ----------
    backbone    : "cnn" | "resnet18"
    num_classes : number of output classes (10 for full MNIST, 2 for binary)
    pretrained  : (ResNet only) whether to load ImageNet weights
    feat_dim    : width of the feature bottleneck
    dropout     : dropout probability

    Returns
    -------
    nn.Module with .get_features(), .classify(), and .forward() methods
    """
    backbone = backbone.lower()
    if backbone == "cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)
    elif backbone == "resnet18":
        return ResNet18DA(
            num_classes=num_classes,
            pretrained=pretrained,
            feat_dim=feat_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose 'cnn' or 'resnet18'.")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(8, 3, 32, 32).to(device)

    for name in ("cnn", "resnet18"):
        model = get_model(name, num_classes=10, pretrained=False).to(device)
        feats  = model.get_features(dummy)
        logits = model.classify(feats)
        print(
            f"[{name:>8}]  input: {tuple(dummy.shape)}"
            f"  ->  features: {tuple(feats.shape)}"
            f"  ->  logits: {tuple(logits.shape)}"
        )

    print("Smoke-test passed.")