import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_DIR = Path("results_cloud/final")
PLOTS_DIR = FINAL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def add_value_labels(bars, suffix=""):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_target_accuracy():
    source = load_json(FINAL_DIR / "source_only_results.json")
    coral = load_json(FINAL_DIR / "deep_coral_results.json")
    upper = load_json(FINAL_DIR / "upper_bound_results.json")

    methods = [
        "Source-only\nCNN",
        "Deep CORAL\nUDA",
        "Target-supervised\nUpper bound",
    ]

    accuracies = [
        source["target_test"]["accuracy"] * 100,
        coral["target_test_after_adaptation"]["accuracy"] * 100,
        upper["target_test"]["accuracy"] * 100,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, accuracies)

    add_value_labels(bars, suffix="%")

    plt.ylabel("Target accuracy (%)")
    plt.title("Target Accuracy under Strong Color Shift")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "target_accuracy_bar.png", dpi=300)
    plt.close()


def plot_error_reduction():
    source = load_json(FINAL_DIR / "source_only_results.json")
    coral = load_json(FINAL_DIR / "deep_coral_results.json")

    source_cm = np.array(source["target_test"]["confusion_matrix"])
    coral_cm = np.array(coral["target_test_after_adaptation"]["confusion_matrix"])

    error_types = ["0-4 → 5-9", "5-9 → 0-4"]

    source_errors = [
        source_cm[0, 1],
        source_cm[1, 0],
    ]

    coral_errors = [
        coral_cm[0, 1],
        coral_cm[1, 0],
    ]

    x = np.arange(len(error_types))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(x - width / 2, source_errors, width, label="Source-only CNN")
    bars2 = plt.bar(x + width / 2, coral_errors, width, label="Deep CORAL")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.xticks(x, error_types)
    plt.ylabel("Number of wrong predictions")
    plt.title("Deep CORAL Reduces Target-Domain Mistakes")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "error_reduction.png", dpi=300)
    plt.close()


def plot_sweep_heatmap():
    csv_path = FINAL_DIR / "sweep_results.csv"
    df = pd.read_csv(csv_path)

    # Convert improvement from decimal to percentage points
    df["delta_pp"] = df["delta"] * 100

    pivot = df.pivot(
        index="target_correlation",
        columns="lambda_coral",
        values="delta_pp",
    )

    plt.figure(figsize=(9, 5))
    image = plt.imshow(pivot.values, aspect="auto")

    plt.colorbar(image, label="Improvement over source-only (percentage points)")

    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=[str(x) for x in pivot.columns],
    )
    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=[str(y) for y in pivot.index],
    )

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            plt.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.xlabel("lambda_coral")
    plt.ylabel("target_correlation")
    plt.title("When Does Deep CORAL Help?")
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / "sweep_heatmap_clean.png", dpi=300)
    plt.close()


def create_final_table_csv():
    source = load_json(FINAL_DIR / "source_only_results.json")
    coral = load_json(FINAL_DIR / "deep_coral_results.json")
    upper = load_json(FINAL_DIR / "upper_bound_results.json")

    source_acc = source["target_test"]["accuracy"] * 100
    coral_acc = coral["target_test_after_adaptation"]["accuracy"] * 100
    upper_acc = upper["target_test"]["accuracy"] * 100

    table = pd.DataFrame(
        [
            {
                "Method": "Source-only CNN",
                "Target labels used": "No",
                "Target accuracy (%)": source_acc,
                "Delta vs source-only (pp)": 0.0,
            },
            {
                "Method": "Deep CORAL",
                "Target labels used": "No",
                "Target accuracy (%)": coral_acc,
                "Delta vs source-only (pp)": coral_acc - source_acc,
            },
            {
                "Method": "Target-supervised upper bound",
                "Target labels used": "Yes",
                "Target accuracy (%)": upper_acc,
                "Delta vs source-only (pp)": upper_acc - source_acc,
            },
        ]
    )

    table.to_csv(FINAL_DIR / "final_results_table.csv", index=False)


def main():
    plot_target_accuracy()
    plot_error_reduction()
    plot_sweep_heatmap()
    create_final_table_csv()

    print("Saved plots to:", PLOTS_DIR)
    print("Saved table to:", FINAL_DIR / "final_results_table.csv")


if __name__ == "__main__":
    main()