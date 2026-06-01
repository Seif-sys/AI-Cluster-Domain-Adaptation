"""
models.py
=========

Defines the CNN model used for all experiments.

Important design:
    get_features(x) returns internal features used by Deep CORAL.
    classify(features) returns logits/classes.
    forward(x) does both.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """
    Small CNN for binary Colored-MNIST.

    Input shape:
        [batch_size, 3, 28, 28]

    Output shape:
        [batch_size, num_classes]

    Feature shape:
        [batch_size, feature_dim]
    """

    def __init__(self, num_classes: int = 2, feature_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()

        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 -> 7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Makes feature map fixed size, even if input size changes later.
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(feature_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return internal feature vectors. Deep CORAL uses these."""
        feature_maps = self.encoder(x)
        features = self.projector(feature_maps)
        return features

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Convert feature vectors into class logits."""
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normal model call: image -> logits."""
        features = self.get_features(x)
        logits = self.classify(features)
        return logits


def get_model(num_classes: int = 2, feature_dim: int = 128, dropout: float = 0.2) -> SmallCNN:
    """Factory function. Useful if we add more model types later."""
    return SmallCNN(num_classes=num_classes, feature_dim=feature_dim, dropout=dropout)


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(8, 3, 28, 28)
    features = model.get_features(x)
    logits = model(x)
    print("features:", features.shape)
    print("logits:", logits.shape)
