import matplotlib.pyplot as plt
import torch

from data_colored_mnist import get_colored_mnist_loaders


def save_debug_images(images, labels, path):
    """
    Save a small image grid so we can visually check the dataset.
    """

    number_of_images = 8

    plt.figure(figsize=(10, 2))

    for i in range(number_of_images):
        image = images[i]

        # PyTorch image shape: [3, 28, 28]
        # Matplotlib wants: [28, 28, 3]
        image = image.permute(1, 2, 0)

        plt.subplot(1, number_of_images, i + 1)
        plt.imshow(image)
        plt.title(f"y={labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    loaders = get_colored_mnist_loaders(batch_size=64)

    source_images, source_labels = next(iter(loaders["source_train"]))
    target_images, target_labels = next(iter(loaders["target_train"]))

    print("Source batch image shape:", source_images.shape)
    print("Source batch label shape:", source_labels.shape)

    print("Target batch image shape:", target_images.shape)
    print("Target batch label shape:", target_labels.shape)

    print("Source labels example:", source_labels[:10])
    print("Target labels example:", target_labels[:10])

    print("Image min value:", torch.min(source_images).item())
    print("Image max value:", torch.max(source_images).item())

    save_debug_images(
        source_images,
        source_labels,
        "results/source_colored_mnist_debug.png",
    )

    save_debug_images(
        target_images,
        target_labels,
        "results/target_colored_mnist_debug.png",
    )

    print("Saved debug images in results/")
    print("Dataset works.")


if __name__ == "__main__":
    main()