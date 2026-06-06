from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_command(command: list[str]) -> None:
    """
    Run one command and stop if it fails.
    """
    print("\n" + "=" * 80)
    print("Running:")
    print(" ".join(command))
    print("=" * 80)

    subprocess.run(command, cwd=ROOT, check=True)


def copy_if_exists(source: Path, target: Path) -> None:
    """
    Copy a file only if it exists.
    """
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"Copied: {source} -> {target}")
    else:
        print(f"Skipped, not found: {source}")


def prepare_final_results_folder() -> None:
    """
    Collect the most important results into results_cloud/final.
    """
    final_dir = ROOT / "results_cloud" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(
        ROOT / "results_cloud" / "source_only" / "results.json",
        final_dir / "source_only_results.json",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "coral" / "results.json",
        final_dir / "deep_coral_results.json",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "upper_bound" / "results.json",
        final_dir / "upper_bound_results.json",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "source_only" / "target_confusion_matrix.png",
        final_dir / "confusion_source_only.png",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "coral" / "target_confusion_matrix.png",
        final_dir / "confusion_deep_coral.png",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "upper_bound" / "target_confusion_matrix.png",
        final_dir / "confusion_upper_bound.png",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "sweep" / "lambda_target_correlation_sweep.csv",
        final_dir / "sweep_results.csv",
    )

    copy_if_exists(
        ROOT / "results_cloud" / "sweep" / "heatmap_delta.png",
        final_dir / "heatmap_delta.png",
    )


def expand_steps(steps: list[str]) -> list[str]:
    """
    Allow shortcuts like:
        core
        analysis
        all
    """
    expanded = []

    for step in steps:
        if step == "core":
            expanded.extend(["source", "coral", "upper", "final"])
        elif step == "analysis":
            expanded.extend(["final", "plots", "pca"])
        elif step == "sweep_all":
            expanded.extend(["sweep", "heatmap", "final", "plots"])
        elif step == "all":
            expanded.extend([
                "source",
                "coral",
                "upper",
                "sweep",
                "heatmap",
                "final",
                "plots",
                "pca",
                "pseudo",
            ])
        else:
            expanded.append(step)

    # Remove duplicates but keep order.
    unique_steps = []
    for step in expanded:
        if step not in unique_steps:
            unique_steps.append(step)

    return unique_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run selected parts of the domain adaptation project."
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=["core"],
        choices=[
            "source",
            "coral",
            "upper",
            "sweep",
            "heatmap",
            "final",
            "plots",
            "pca",
            "pseudo",
            "core",
            "analysis",
            "sweep_all",
            "all",
        ],
        help="Which parts to run.",
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sweep_epochs", type=int, default=3)

    parser.add_argument("--target_correlation", type=float, default=0.10)
    parser.add_argument("--lambda_coral", type=float, default=10.0)

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 5.0, 10.0, 25.0],
    )

    parser.add_argument(
        "--target_correlations",
        nargs="+",
        type=float,
        default=[0.10, 0.30, 0.50],
    )

    parser.add_argument(
        "--source_checkpoint",
        type=str,
        default="checkpoints/source_only_best.pt",
    )

    parser.add_argument(
        "--coral_checkpoint",
        type=str,
        default="checkpoints/coral_best.pt",
    )

    parser.add_argument("--confidence_threshold", type=float, default=0.95)

    args = parser.parse_args()
    steps = expand_steps(args.steps)

    print("Selected steps:", steps)

    for step in steps:
        if step == "source":
            run_command([
                PYTHON,
                "src/train_source.py",
                "--epochs",
                str(args.epochs),
                "--target_correlation",
                str(args.target_correlation),
            ])

        elif step == "coral":
            run_command([
                PYTHON,
                "src/train_coral.py",
                "--epochs",
                str(args.epochs),
                "--target_correlation",
                str(args.target_correlation),
                "--lambda_coral",
                str(args.lambda_coral),
            ])

        elif step == "upper":
            run_command([
                PYTHON,
                "src/train_upper_bound.py",
                "--epochs",
                str(args.epochs),
                "--target_correlation",
                str(args.target_correlation),
            ])

        elif step == "sweep":
            run_command([
                PYTHON,
                "src/sweep_lambdas.py",
                "--epochs",
                str(args.sweep_epochs),
                "--lambdas",
                *[str(x) for x in args.lambdas],
                "--target_correlations",
                *[str(x) for x in args.target_correlations],
            ])

        elif step == "heatmap":
            run_command([
                PYTHON,
                "src/plot_sweep_heatmap.py",
                "--value",
                "delta",
            ])

        elif step == "final":
            prepare_final_results_folder()

        elif step == "plots":
            run_command([
                PYTHON,
                "src/make_presentation_plots.py",
            ])

        elif step == "pca":
            run_command([
                PYTHON,
                "src/plot_pca_features.py",
                "--source_checkpoint",
                args.source_checkpoint,
                "--coral_checkpoint",
                args.coral_checkpoint,
                "--target_correlation",
                str(args.target_correlation),
            ])

        elif step == "pseudo":
            run_command([
                PYTHON,
                "src/train_pseudo_label.py",
                "--source_checkpoint",
                args.source_checkpoint,
                "--epochs",
                str(args.epochs),
                "--target_correlation",
                str(args.target_correlation),
                "--confidence_threshold",
                str(args.confidence_threshold),
            ])

    print("\nDone.")


if __name__ == "__main__":
    main()