"""
plot_full_sweep.py
==================

Create useful presentation plots from results/full_sweep/summary.csv.

Example:
    python src/plot_full_sweep.py --summary_csv results/full_sweep/summary.csv --output_dir results/full_sweep/plots
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        for key in row:
            try:
                row[key] = float(row[key])
            except ValueError:
                pass

    return rows


def unique_sorted(rows: list[dict], key: str) -> list:
    return sorted(set(row[key] for row in rows))


def save_heatmap(
    rows: list[dict],
    value_key: str,
    title: str,
    output_path: Path,
    fmt: str = ".3f",
) -> None:
    feature_dims = unique_sorted(rows, "feature_dim")
    lambdas = unique_sorted(rows, "lambda_coral")

    values = []
    for feature_dim in feature_dims:
        row_values = []
        for lambda_coral in lambdas:
            matching = [
                row for row in rows
                if row["feature_dim"] == feature_dim and row["lambda_coral"] == lambda_coral
            ]
            row_values.append(matching[0][value_key] if matching else 0.0)
        values.append(row_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, aspect="auto")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("lambda_coral")
    ax.set_ylabel("feature_dim")

    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([str(x) for x in lambdas])

    ax.set_yticks(range(len(feature_dims)))
    ax.set_yticklabels([str(int(x)) for x in feature_dims])

    for i, feature_dim in enumerate(feature_dims):
        for j, lambda_coral in enumerate(lambdas):
            ax.text(j, i, format(values[i][j], fmt), ha="center", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)


def save_best_method_plot(rows: list[dict], output_path: Path) -> None:
    feature_dims = unique_sorted(rows, "feature_dim")

    source_values = []
    pseudo_values = []
    upper_values = []
    best_coral_values = []

    for feature_dim in feature_dims:
        feature_rows = [row for row in rows if row["feature_dim"] == feature_dim]

        source_values.append(feature_rows[0]["source_only_target_acc"])
        pseudo_values.append(feature_rows[0]["pseudo_label_target_acc"])
        upper_values.append(feature_rows[0]["upper_bound_target_acc"])
        best_coral_values.append(max(row["coral_target_acc"] for row in feature_rows))

    plt.figure(figsize=(8, 5))
    plt.plot(feature_dims, source_values, marker="o", label="Source-only")
    plt.plot(feature_dims, best_coral_values, marker="o", label="Best Deep CORAL")
    plt.plot(feature_dims, pseudo_values, marker="o", label="Pseudo-labeling")
    plt.plot(feature_dims, upper_values, marker="o", label="Upper bound")

    plt.title("Target accuracy by feature dimension")
    plt.xlabel("feature_dim")
    plt.ylabel("target accuracy")
    plt.legend()
    plt.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)


def save_best_lambda_plot(rows: list[dict], output_path: Path) -> None:
    feature_dims = unique_sorted(rows, "feature_dim")

    best_lambdas = []
    best_accuracies = []

    for feature_dim in feature_dims:
        feature_rows = [row for row in rows if row["feature_dim"] == feature_dim]
        best_row = max(feature_rows, key=lambda row: row["coral_target_acc"])
        best_lambdas.append(best_row["lambda_coral"])
        best_accuracies.append(best_row["coral_target_acc"])

    plt.figure(figsize=(8, 5))
    plt.plot(feature_dims, best_lambdas, marker="o")

    for x, y, acc in zip(feature_dims, best_lambdas, best_accuracies):
        plt.text(x, y, f"acc={acc:.3f}", ha="center", va="bottom")

    plt.title("Best CORAL lambda per feature dimension")
    plt.xlabel("feature_dim")
    plt.ylabel("best lambda_coral")
    plt.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)


def save_training_time_plot(rows: list[dict], output_path: Path) -> None:
    feature_dims = unique_sorted(rows, "feature_dim")

    source_times = []
    pseudo_times = []
    upper_times = []
    best_coral_times = []

    for feature_dim in feature_dims:
        feature_rows = [row for row in rows if row["feature_dim"] == feature_dim]

        source_times.append(feature_rows[0]["source_only_time"])
        pseudo_times.append(feature_rows[0]["pseudo_label_time"])
        upper_times.append(feature_rows[0]["upper_bound_time"])

        best_row = max(feature_rows, key=lambda row: row["coral_target_acc"])
        best_coral_times.append(best_row["coral_time"])

    plt.figure(figsize=(8, 5))
    plt.plot(feature_dims, source_times, marker="o", label="Source-only")
    plt.plot(feature_dims, best_coral_times, marker="o", label="Best Deep CORAL")
    plt.plot(feature_dims, pseudo_times, marker="o", label="Pseudo-labeling")
    plt.plot(feature_dims, upper_times, marker="o", label="Upper bound")

    plt.title("Training time by feature dimension")
    plt.xlabel("feature_dim")
    plt.ylabel("seconds")
    plt.legend()
    plt.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Saved:", output_path)


def save_takeaways(rows: list[dict], output_path: Path) -> None:
    best_coral = max(rows, key=lambda row: row["coral_target_acc"])
    best_delta = max(rows, key=lambda row: row["coral_delta_vs_source"])
    worst_delta = min(rows, key=lambda row: row["coral_delta_vs_source"])

    feature_groups = defaultdict(list)
    for row in rows:
        feature_groups[row["feature_dim"]].append(row)

    lines = []

    lines.append("FULL SWEEP TAKEAWAYS")
    lines.append("====================")
    lines.append("")

    lines.append("Best Deep CORAL target accuracy:")
    lines.append(
        f"  feature_dim={int(best_coral['feature_dim'])}, "
        f"lambda={best_coral['lambda_coral']}, "
        f"target_acc={best_coral['coral_target_acc']:.4f}"
    )
    lines.append("")

    lines.append("Best Deep CORAL improvement over source-only:")
    lines.append(
        f"  feature_dim={int(best_delta['feature_dim'])}, "
        f"lambda={best_delta['lambda_coral']}, "
        f"delta={best_delta['coral_delta_vs_source']:.4f}"
    )
    lines.append("")

    lines.append("Worst Deep CORAL improvement over source-only:")
    lines.append(
        f"  feature_dim={int(worst_delta['feature_dim'])}, "
        f"lambda={worst_delta['lambda_coral']}, "
        f"delta={worst_delta['coral_delta_vs_source']:.4f}"
    )
    lines.append("")

    lines.append("Best method per feature dimension:")
    for feature_dim, feature_rows in sorted(feature_groups.items()):
        best_coral_row = max(feature_rows, key=lambda row: row["coral_target_acc"])

        source_acc = feature_rows[0]["source_only_target_acc"]
        pseudo_acc = feature_rows[0]["pseudo_label_target_acc"]
        upper_acc = feature_rows[0]["upper_bound_target_acc"]
        coral_acc = best_coral_row["coral_target_acc"]

        methods = {
            "source-only": source_acc,
            "best Deep CORAL": coral_acc,
            "pseudo-labeling": pseudo_acc,
            "upper bound": upper_acc,
        }

        best_method = max(methods, key=methods.get)

        lines.append(
            f"  feature_dim={int(feature_dim)}: "
            f"best realistic method={best_method}, "
            f"source={source_acc:.4f}, "
            f"coral={coral_acc:.4f}, "
            f"pseudo={pseudo_acc:.4f}, "
            f"upper={upper_acc:.4f}, "
            f"best_lambda={best_coral_row['lambda_coral']}"
        )

    lines.append("")
    lines.append("How to use this in presentation:")
    lines.append("  - Source-only is the baseline.")
    lines.append("  - Upper bound is the best-case reference using target labels.")
    lines.append("  - Deep CORAL is useful if coral_delta_vs_source is positive.")
    lines.append("  - Pseudo-labeling is useful if pseudo_delta_vs_source is positive.")
    lines.append("  - gap_to_upper shows how far each method is from the ideal target-supervised model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print("Saved:", output_path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="results/full_sweep/summary.csv")
    parser.add_argument("--output_dir", type=str, default="results/full_sweep/plots")
    return parser.parse_args()


def main() -> None:
    args = get_args()

    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir)

    rows = read_rows(summary_csv)

    save_heatmap(
        rows,
        value_key="coral_target_acc",
        title="Deep CORAL target accuracy",
        output_path=output_dir / "heatmap_coral_target_accuracy.png",
    )

    save_heatmap(
        rows,
        value_key="coral_delta_vs_source",
        title="Deep CORAL improvement over source-only",
        output_path=output_dir / "heatmap_coral_delta_vs_source.png",
    )

    save_heatmap(
        rows,
        value_key="coral_gap_to_upper",
        title="Deep CORAL gap to upper bound",
        output_path=output_dir / "heatmap_coral_gap_to_upper.png",
    )

    save_best_method_plot(
        rows,
        output_path=output_dir / "target_accuracy_by_feature_dim.png",
    )

    save_best_lambda_plot(
        rows,
        output_path=output_dir / "best_lambda_by_feature_dim.png",
    )

    save_training_time_plot(
        rows,
        output_path=output_dir / "training_time_by_feature_dim.png",
    )

    save_takeaways(
        rows,
        output_path=output_dir / "presentation_takeaways.txt",
    )

    print("\nPlots done.")


if __name__ == "__main__":
    main()