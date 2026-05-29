import torch

from models import SmallCNN


def main():
    model = SmallCNN(num_classes=2)

    fake_images = torch.randn(64, 3, 28, 28)

    logits = model(fake_images)
    print("Logits shape:", logits.shape)

    logits, features = model(fake_images, return_features=True)
    print("Logits shape with features:", logits.shape)
    print("Features shape:", features.shape)

    print("Model works.")


if __name__ == "__main__":
    main()