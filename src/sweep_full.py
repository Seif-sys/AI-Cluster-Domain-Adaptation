"""
sweep_full.py
=============

Full experiment sweep.

Runs:
    - source-only
    - upper-bound
    - pseudo-labeling
    - Deep CORAL with multiple lambda values

For multiple feature dimensions.

Example quick test:
    python src/sweep_full.py --epochs 1 --feature_dims 64 --lambdas 1.0 --max_train_samples 1000 --make_plots --clean

Example full run:
    python src/sweep_full.py --epochs 5 --feature_dims 64 128 256 512 --lambdas 0.1 1.0 5.0 10.0 25.0 --make_plots --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def safe_name(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def run_command(command: list[str], skip_if_exists: Path | None = None) -> None:
    if skip_if_exists is not None and skip_if_exists.exists():
        print(f"\nSkipping existing result: {skip_if_exists}")
        return

    print("\n" + "=" * 100)
    print("Running:")
    print(" ".join(command))
    print("=" * 100)

    subprocess.run(command, cwd=ROOT, check=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def accuracy(result: dict, key: str) -> float:
    return float(result[key]["accuracy"])


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        raise RuntimeError("No rows to write.")

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full feature/lambda sweep")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--source_correlation", type=float, default=0.99)
    parser.add_argument("--target_correlation", type=float, default=0.10)

    parser.add_argument(
        "--feature_dims",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512],
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 5.0, 10.0, 25.0],
    )

    parser.add_argument("--confidence_threshold", type=float, default=0.95)
    parser.add_argument("--pseudo_weight", type=float, default=1.0)

    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--make_plots", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = get_args()

    results_root = ROOT / "results" / "full_sweep"
    checkpoints_root = ROOT / "checkpoints" / "full_sweep"

    if args.clean:
        if results_root.exists():
            shutil.rmtree(results_root)
        if checkpoints_root.exists():
            shutil.rmtree(checkpoints_root)

    results_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for feature_dim in args.feature_dims:
        print("\n\n")
        print("#" * 100)
        print(f"FEATURE DIMENSION: {feature_dim}")
        print("#" * 100)

        feature_results_dir = results_root / f"feature_dim_{feature_dim}"
        feature_checkpoint_dir = checkpoints_root / f"feature_dim_{feature_dim}"

        feature_results_dir.mkdir(parents=True, exist_ok=True)
        feature_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        source_checkpoint = feature_checkpoint_dir / "source_only_best.pt"
        upper_checkpoint = feature_checkpoint_dir / "upper_bound_best.pt"
        pseudo_checkpoint = feature_checkpoint_dir / "pseudo_label_best.pt"

        source_output = feature_results_dir / "source_only"
        upper_output = feature_results_dir / "upper_bound"
        pseudo_output = feature_results_dir / "pseudo_label"

        # 1. Source-only
        source_command = [
            PYTHON,
            "src/train_source.py",
            "--checkpoint_path", str(source_checkpoint),
            "--output_dir", str(source_output),
        ]
        source_command = add_common_args(source_command, args, feature_dim)

        run_command(
            source_command,
            skip_if_exists=(source_output / "results.json") if args.skip_existing else None,
        )

        # 2. Upper bound
        upper_command = [
            PYTHON,
            "src/train_upper_bound.py",
            "--source_checkpoint", str(source_checkpoint),
            "--checkpoint_path", str(upper_checkpoint),
            "--output_dir", str(upper_output),
        ]
        upper_command = add_common_args(upper_command, args, feature_dim)

        run_command(
            upper_command,
            skip_if_exists=(upper_output / "results.json") if args.skip_existing else None,
        )

        # 3. Pseudo-labeling
        pseudo_command = [
            PYTHON,
            "src/train_pseudo_label.py",
            "--source_checkpoint", str(source_checkpoint),
            "--checkpoint_path", str(pseudo_checkpoint),
            "--output_dir", str(pseudo_output),
            "--confidence_threshold", str(args.confidence_threshold),
            "--pseudo_weight", str(args.pseudo_weight),
        ]
        pseudo_command = add_common_args(pseudo_command, args, feature_dim)

        run_command(
            pseudo_command,
            skip_if_exists=(pseudo_output / "results.json") if args.skip_existing else None,
        )

        source_results = read_json(source_output / "results.json")
        upper_results = read_json(upper_output / "results.json")
        pseudo_results = read_json(pseudo_output / "results.json")

        source_target_acc = accuracy(source_results, "target_test")
        source_source_acc = accuracy(source_results, "source_test")

        upper_target_acc = accuracy(upper_results, "target_test")
        upper_source_acc = accuracy(upper_results, "source_test")

        pseudo_target_acc = accuracy(pseudo_results, "target_test")
        pseudo_source_acc = accuracy(pseudo_results, "source_test")

        # 4. CORAL for every lambda
        for lambda_coral in args.lambdas:
            lambda_name = safe_name(lambda_coral)

            coral_checkpoint = feature_checkpoint_dir / f"coral_lambda_{lambda_name}_best.pt"
            coral_output = feature_results_dir / f"coral_lambda_{lambda_name}"

            coral_command = [
                PYTHON,
                "src/train_coral.py",
                "--source_checkpoint", str(source_checkpoint),
                "--checkpoint_path", str(coral_checkpoint),
                "--output_dir", str(coral_output),
                "--lambda_coral", str(lambda_coral),
            ]
            coral_command = add_common_args(coral_command, args, feature_dim)

            run_command(
                coral_command,
                skip_if_exists=(coral_output / "results.json") if args.skip_existing else None,
            )

            coral_results = read_json(coral_output / "results.json")

            coral_target_acc = accuracy(coral_results, "target_test_after_adaptation")
            coral_source_acc = accuracy(coral_results, "source_test_after_adaptation")
            coral_before_acc = accuracy(coral_results, "target_before_adaptation")

            summary_rows.append(
                {
                    "feature_dim": feature_dim,
                    "lambda_coral": lambda_coral,

                    "source_only_source_acc": source_source_acc,
                    "source_only_target_acc": source_target_acc,

                    "coral_source_acc": coral_source_acc,
                    "coral_target_before_acc": coral_before_acc,
                    "coral_target_acc": coral_target_acc,

                    "pseudo_label_source_acc": pseudo_source_acc,
                    "pseudo_label_target_acc": pseudo_target_acc,

                    "upper_bound_source_acc": upper_source_acc,
                    "upper_bound_target_acc": upper_target_acc,

                    "coral_delta_vs_source": coral_target_acc - source_target_acc,
                    "pseudo_delta_vs_source": pseudo_target_acc - source_target_acc,

                    "coral_gap_to_upper": upper_target_acc - coral_target_acc,
                    "pseudo_gap_to_upper": upper_target_acc - pseudo_target_acc,
                    "source_gap_to_upper": upper_target_acc - source_target_acc,

                    "source_only_time": source_results["training_time_seconds"],
                    "coral_time": coral_results["training_time_seconds"],
                    "pseudo_label_time": pseudo_results["training_time_seconds"],
                    "upper_bound_time": upper_results["training_time_seconds"],

                    "source_output_dir": str(source_output),
                    "coral_output_dir": str(coral_output),
                    "pseudo_output_dir": str(pseudo_output),
                    "upper_output_dir": str(upper_output),
                }
            )

    summary_path = results_root / "summary.csv"
    write_csv(summary_path, summary_rows)

    if args.make_plots:
        run_command([
            PYTHON,
            "src/plot_full_sweep.py",
            "--summary_csv", str(summary_path),
            "--output_dir", str(results_root / "plots"),
        ])

    print("\nFull sweep done.")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()