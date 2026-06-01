"""
plot_sweep_heatmap.py
=====================

Create a 2D heatmap from the sweep CSV.

x-axis: lambda_coral
y-axis: target_correlation
value: delta = CORAL target accuracy - source-only target accuracy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Plot lambda sweep heatmap")
    parser.add_argument("--csv", type=str, default="results/sweep/lambda_target_correlation_sweep.csv")
    parser.add_argument("--value", type=str, default="delta", choices=["delta", "coral_target_acc", "source_only_target_acc"])
    parser.add_argument("--output", type=str, default="results/sweep/heatmap_delta.png")
    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.csv)

    lambdas = sorted(df["lambda_coral"].unique())
    target_corrs = sorted(df["target_correlation"].unique())

    matrix = np.zeros((len(target_corrs), len(lambdas)))

    for i, target_corr in enumerate(target_corrs):
        for j, lambda_coral in enumerate(lambdas):
            row = df[(df["target_correlation"] == target_corr) & (df["lambda_coral"] == lambda_coral)]
            matrix[i, j] = row[args.value].iloc[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=args.value)
    plt.xticks(range(len(lambdas)), lambdas)
    plt.yticks(range(len(target_corrs)), target_corrs)
    plt.xlabel("lambda_coral")
    plt.ylabel("target_correlation")
    plt.title(f"Sweep heatmap: {args.value}")

    for i in range(len(target_corrs)):
        for j in range(len(lambdas)):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print("Saved heatmap to", output)


if __name__ == "__main__":
    main()
