from __future__ import annotations

import torch

from models import get_model


def main() -> None:
    model = get_model(num_classes=2, feature_dim=128)
    model.eval()

    fake_images = torch.randn(64, 3, 28, 28)

    with torch.no_grad():
        logits = model(fake_images)
        features = model.get_features(fake_images)

    print("Logits shape:", logits.shape)
    print("Features shape:", features.shape)

    assert logits.shape == (64, 2)
    assert features.shape == (64, 128)

    print("Model works.")


if __name__ == "__main__":
    main()