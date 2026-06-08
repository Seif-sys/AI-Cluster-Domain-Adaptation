"""
sweep_feature_dims.py
=====================

Run source-only, Deep CORAL, and upper-bound experiments
for different feature dimensions.

Example:
    python src/sweep_feature_dims.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100)

    subprocess.run(command, check=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep over feature dimensions")

    parser.add_argument("--feature_dims", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)
    parser.add_argument("--lambda_coral", type=float, default=1.0)

    parser.add_argument("--max_train_samples", type=int, default=None)

    return parser.parse_args()


def add_common_args(command: list[str], args: argparse.Namespace, feature_dim: int) -> list[str]:
    command += [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--source_correlation", str(args.source_correlation),
        "--target_correlation", str(args.target_correlation),
        "--feature_dim", str(feature_dim),
    ]

    if args.max_train_samples is not None:
        command += ["--max_train_samples", str(args.max_train_samples)]

    return command


def main() -> None:
    args = get_args()

    base_results_dir = Path("results") / "feature_dim_sweep"
    base_checkpoint_dir = Path("checkpoints") / "feature_dim_sweep"

    base_results_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for feature_dim in args.feature_dims:
        print("\n\n")
        print("#" * 100)
        print(f"FEATURE DIMENSION: {feature_dim}")
        print("#" * 100)

        run_results_dir = base_results_dir / f"feature_dim_{feature_dim}"
        run_checkpoint_dir = base_checkpoint_dir / f"feature_dim_{feature_dim}"

        run_results_dir.mkdir(parents=True, exist_ok=True)
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        source_checkpoint = run_checkpoint_dir / "source_only_best.pt"
        coral_checkpoint = run_checkpoint_dir / "coral_best.pt"
        upper_checkpoint = run_checkpoint_dir / "upper_bound_best.pt"
        pseudo_checkpoint = run_checkpoint_dir / "pseudo_label_best.pt"

        source_output = run_results_dir / "source_only"
        coral_output = run_results_dir / "coral"
        upper_output = run_results_dir / "upper_bound"
        pseudo_output = run_results_dir / "pseudo_label"

        # 1. Source-only
        source_command = [
            sys.executable,
            "src/train_source.py",
            "--checkpoint_path", str(source_checkpoint),
            "--output_dir", str(source_output),
        ]
        source_command = add_common_args(source_command, args, feature_dim)
        run_command(source_command)

        # 2. Deep CORAL
        coral_command = [
            sys.executable,
            "src/train_coral.py",
            "--source_checkpoint", str(source_checkpoint),
            "--checkpoint_path", str(coral_checkpoint),
            "--output_dir", str(coral_output),
            "--lambda_coral", str(args.lambda_coral),
        ]
        coral_command = add_common_args(coral_command, args, feature_dim)
        run_command(coral_command)

        # 3. Upper bound
        upper_command = [
            sys.executable,
            "src/train_upper_bound.py",
            "--source_checkpoint", str(source_checkpoint),
            "--checkpoint_path", str(upper_checkpoint),
            "--output_dir", str(upper_output),
        ]
        upper_command = add_common_args(upper_command, args, feature_dim)
        run_command(upper_command)
        
        # 4. Pseudo-labeling
        pseudo_command = [
            sys.executable,
            "src/train_pseudo_label.py",
            "--source_checkpoint", str(source_checkpoint),
            "--checkpoint_path", str(pseudo_checkpoint),
            "--output_dir", str(pseudo_output),
            "--confidence_threshold", "0.95",
            "--pseudo_weight", "1.0",
        ]
        pseudo_command = add_common_args(pseudo_command, args, feature_dim)
        run_command(pseudo_command)

        # Read results
        source_results = read_json(source_output / "results.json")
        coral_results = read_json(coral_output / "results.json")
        upper_results = read_json(upper_output / "results.json")
        pseudo_results = read_json(pseudo_output / "results.json")

        summary_rows.append(
            {
                "feature_dim": feature_dim,
                "source_only_target_acc": source_results["target_test"]["accuracy"],
                "coral_target_acc": coral_results["target_test_after_adaptation"]["accuracy"],
                "upper_bound_target_acc": upper_results["target_test"]["accuracy"],
                "pseudo_label_target_acc": pseudo_results["target_test"]["accuracy"],
                "source_only_time": source_results["training_time_seconds"],
                "coral_time": coral_results["training_time_seconds"],
                "upper_bound_time": upper_results["training_time_seconds"],
                "pseudo_label_time": pseudo_results["training_time_seconds"],
            }
        )

    summary_path = base_results_dir / "feature_dim_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    main()