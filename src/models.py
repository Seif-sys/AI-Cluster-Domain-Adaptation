import torch
from torch import nn


class SmallCNN(nn.Module):
    """
    Small CNN for binary Colored-MNIST classification.

    Input:
        image tensor with shape [batch_size, 3, 28, 28]

    Output:
        logits with shape [batch_size, 2]

    The model can also return internal features for Deep CORAL.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        self.feature_layer = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feature_maps = self.conv_layers(x)

        flattened = self.flatten(feature_maps)

        features = self.feature_layer(flattened)
        features = self.relu(features)

        logits = self.output_layer(features)

        if return_features:
            return logits, features

        return logits