"""
sweep_lambdas.py
================

Run a small 2D sweep:
    x-axis: lambda_coral
    y-axis: target_correlation

This helps answer:
    Does CORAL work only for some lambda values?
    Does CORAL behave differently when the target shift is stronger/weaker?

Important scientific note:
    If you use target test accuracy to choose lambda, that is tuning on the test set.
    For the report, present this as an ablation/sensitivity analysis and show all values.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from utils import load_json


def get_args():
    parser = argparse.ArgumentParser(description="Run lambda/target-correlation sweep")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 1.0, 5.0, 10.0, 25.0, 50.0])
    parser.add_argument("--target_correlations", nargs="+", type=float, default=[0.10, 0.30, 0.50])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default="results/sweep/lambda_target_correlation_sweep.csv")
    return parser.parse_args()


def run_command(command):
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    args = get_args()
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Source-only baseline must exist first for every target correlation,
    # because the baseline target accuracy changes when target correlation changes.
    for target_corr in args.target_correlations:
        source_out = Path(f"results/sweep/source_tc_{target_corr}")
        source_ckpt = Path(f"checkpoints/sweep/source_tc_{target_corr}.pt")

        command = [
            sys.executable, "src/train_source.py",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--source_correlation", str(args.source_correlation),
            "--target_correlation", str(target_corr),
            "--output_dir", str(source_out),
            "--checkpoint_path", str(source_ckpt),
        ]
        if args.max_train_samples is not None:
            command += ["--max_train_samples", str(args.max_train_samples)]

        run_command(command)
        source_results = load_json(source_out / "results.json")
        source_target_acc = source_results["target_test"]["accuracy"]

        for lambda_coral in args.lambdas:
            coral_out = Path(f"results/sweep/coral_tc_{target_corr}_lambda_{lambda_coral}")
            coral_ckpt = Path(f"checkpoints/sweep/coral_tc_{target_corr}_lambda_{lambda_coral}.pt")

            command = [
                sys.executable, "src/train_coral.py",
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--seed", str(args.seed),
                "--source_correlation", str(args.source_correlation),
                "--target_correlation", str(target_corr),
                "--lambda_coral", str(lambda_coral),
                "--source_checkpoint", str(source_ckpt),
                "--output_dir", str(coral_out),
                "--checkpoint_path", str(coral_ckpt),
            ]
            if args.max_train_samples is not None:
                command += ["--max_train_samples", str(args.max_train_samples)]

            run_command(command)
            coral_results = load_json(coral_out / "results.json")
            coral_target_acc = coral_results["target_test_after_adaptation"]["accuracy"]
            delta = coral_target_acc - source_target_acc

            rows.append({
                "target_correlation": target_corr,
                "lambda_coral": lambda_coral,
                "source_only_target_acc": source_target_acc,
                "coral_target_acc": coral_target_acc,
                "delta": delta,
                "negative_transfer": delta < 0,
            })

            with open(output_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    print("Saved sweep CSV to", output_csv)


if __name__ == "__main__":
    main()
