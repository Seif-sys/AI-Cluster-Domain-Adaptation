import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from data_colored_mnist import get_colored_mnist_loaders
from models import SmallCNN
from utils import get_device, set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


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
        train_loss, train_acc = train_one_epoch(
            model,
            loaders["target_train"],
            criterion,
            optimizer,
            device,
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
            f"target test acc: {target_test_acc:.4f}"
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
        },
        "checkpoints/target_supervised_upper_bound_cnn.pt",
    )

    results = {
        "method": "target_supervised_upper_bound_cnn",
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "source_correlation": source_correlation,
        "target_correlation": target_correlation,
        "training_time_seconds": training_time,
        "history": history,
    }

    with open("results/target_supervised_upper_bound_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Saved checkpoint to checkpoints/target_supervised_upper_bound_cnn.pt")
    print("Saved results to results/target_supervised_upper_bound_results.json")
    print("Target-supervised upper bound training done.")


if __name__ == "__main__":
    main()