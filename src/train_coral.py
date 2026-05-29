import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_colored_mnist import get_colored_mnist_loaders
from models import SmallCNN
from utils import get_device, set_seed


def coral_loss(source_features, target_features):
    """
    Deep CORAL loss.

    It compares the covariance/statistics of source features
    and target features.
    """

    source_features = source_features - source_features.mean(dim=0)
    target_features = target_features - target_features.mean(dim=0)

    source_cov = source_features.T @ source_features
    target_cov = target_features.T @ target_features

    source_cov = source_cov / (source_features.size(0) - 1)
    target_cov = target_cov / (target_features.size(0) - 1)

    feature_dim = source_features.size(1)

    loss = torch.mean((source_cov - target_cov) ** 2)
    loss = loss / (4 * feature_dim * feature_dim)

    return loss


def train_one_epoch_coral(
    model,
    source_loader,
    target_loader,
    criterion,
    optimizer,
    device,
    coral_weight,
):
    model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_coral_loss = 0.0
    correct = 0
    total = 0

    for (source_images, source_labels), (target_images, _) in zip(source_loader, target_loader):
        source_images = source_images.to(device)
        source_labels = source_labels.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        source_logits, source_features = model(source_images, return_features=True)
        _, target_features = model(target_images, return_features=True)

        classification_loss = criterion(source_logits, source_labels)
        adaptation_loss = coral_loss(source_features, target_features)

        loss = classification_loss + coral_weight * adaptation_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * source_labels.size(0)
        total_classification_loss += classification_loss.item() * source_labels.size(0)
        total_coral_loss += adaptation_loss.item() * source_labels.size(0)

        predictions = torch.argmax(source_logits, dim=1)
        correct += (predictions == source_labels).sum().item()
        total += source_labels.size(0)

    average_loss = total_loss / total
    average_classification_loss = total_classification_loss / total
    average_coral_loss = total_coral_loss / total
    accuracy = correct / total

    return average_loss, average_classification_loss, average_coral_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    seed = 42
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    source_correlation = 0.99
    target_correlation = 0.50
    coral_weight = 1.0

    set_seed(seed)
    device = get_device()

    print("Device:", device)

    loaders = get_colored_mnist_loaders(
        batch_size=batch_size,
        source_correlation=source_correlation,
        target_correlation=target_correlation,
        seed=seed,
    )

    model = SmallCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, class_loss, coral_train_loss, train_acc = train_one_epoch_coral(
            model=model,
            source_loader=loaders["source_train"],
            target_loader=loaders["target_train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            coral_weight=coral_weight,
        )

        source_test_loss, source_test_acc = evaluate(
            model,
            loaders["source_test"],
            criterion,
            device,
        )

        target_test_loss, target_test_acc = evaluate(
            model,
            loaders["target_test"],
            criterion,
            device,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "classification_loss": class_loss,
            "coral_loss": coral_train_loss,
            "train_acc": train_acc,
            "source_test_loss": source_test_loss,
            "source_test_acc": source_test_acc,
            "target_test_loss": target_test_loss,
            "target_test_acc": target_test_acc,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train acc: {train_acc:.4f} | "
            f"source test acc: {source_test_acc:.4f} | "
            f"target test acc: {target_test_acc:.4f} | "
            f"coral loss: {coral_train_loss:.6f}"
        )

    training_time = time.time() - start_time

    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "source_correlation": source_correlation,
            "target_correlation": target_correlation,
            "coral_weight": coral_weight,
        },
        "checkpoints/deep_coral_cnn.pt",
    )

    results = {
        "method": "deep_coral_cnn",
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "source_correlation": source_correlation,
        "target_correlation": target_correlation,
        "coral_weight": coral_weight,
        "training_time_seconds": training_time,
        "history": history,
    }

    with open("results/deep_coral_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Saved checkpoint to checkpoints/deep_coral_cnn.pt")
    print("Saved results to results/deep_coral_results.json")
    print("Deep CORAL training done.")


if __name__ == "__main__":
    main()