"""
plot_results.py
===============
Generates all experiment result graphs from the domain adaptation project.

Reads metrics.csv files from your output directories and falls back to
hardcoded results from the conversation if a file is missing.

Usage
-----
python plot_results.py

Output
------
All plots saved to ./plots/ as high-resolution PNGs.

Directory structure expected
----------------------------
./results/source/          metrics.csv  (source-only baseline, colored mnist)
./results/source_rot/      metrics.csv  (source-only baseline, rotated mnist)
./outputs/coral_rot/       metrics.csv  (CORAL lambda=1.0, rotated)
./outputs/coral_rotl10/    metrics.csv  (CORAL lambda=10.0, rotated)
./outputs/coral_rotl25/    metrics.csv  (CORAL lambda=25.0, rotated)
./outputs/coral_rot_l1/    metrics.csv  (CORAL lambda=1.0, colored)
./outputs/upper_bound_rot/ metrics.csv  (upper bound, rotated)
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output directory ──────────────────────────────────────────────────────────
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})

COLORS = {
    "baseline": "#64748B",
    "coral_l1":  "#0D9488",
    "coral_l10": "#F59E0B",
    "coral_l25": "#0D9488",
    "upper":     "#22C55E",
    "source":    "#065A82",
    "target":    "#F96167",
    "val":       "#A855F7",
    "ce":        "#0D9488",
    "coral_loss":"#F96167",
}

# ── Hardcoded final results (from experiment logs) ────────────────────────────
# Used for summary plots and as fallback if CSVs are missing.

COLORED_RESULTS = {
    "labels":   ["Source-only\nbaseline", "CORAL\nλ=1.0", "CORAL\nλ=10.0", "CORAL\nλ=25.0", "Upper bound\n(supervised)"],
    "target":   [89.71, 88.45, 89.91, 88.52, 99.07],
    "source":   [99.81, 99.80, 99.81, 99.81, 99.29],
    "delta":    [0.0,   -1.26,  0.20, -1.19,  9.36],
    "colors":   [COLORS["baseline"], COLORS["target"], COLORS["coral_l10"],
                 COLORS["target"], COLORS["upper"]],
}

ROTATED_RESULTS = {
    "labels":   ["Source-only\nbaseline", "CORAL\nλ=1.0", "CORAL\nλ=10.0", "CORAL\nλ=25.0", "Upper bound\n(supervised)"],
    "target":   [65.52, 70.22, 68.03, 70.31, 99.19],
    "source":   [99.44, 99.50, 99.50, 99.50, 85.29],
    "delta":    [0.0,    4.70,  2.51,  4.79, 33.67],
    "colors":   [COLORS["baseline"], COLORS["coral_l1"], COLORS["coral_l10"],
                 COLORS["coral_l25"], COLORS["upper"]],
}

# Per-class accuracy: source-only vs best CORAL (rotated mnist, lambda=25)
ROTATED_PERCLASS_BEFORE = [99.2, 82.2, 46.3, 77.3, 35.0, 57.5, 71.0, 36.9, 68.5, 79.8]
ROTATED_PERCLASS_CORAL  = [96.7, 93.7, 58.0, 80.7, 44.6, 63.5, 66.1, 50.5, 72.6, 76.7]
ROTATED_PERCLASS_UPPER  = [99.7, 99.8, 99.3, 99.6, 99.3, 99.0, 98.1, 99.0, 99.5, 98.4]

COLORED_PERCLASS_BEFORE = [88.2, 88.7]  # binary: class 0 (digits 0-4), class 1 (digits 5-9)
COLORED_PERCLASS_CORAL  = [89.3, 90.6]  # best: lambda=10
COLORED_PERCLASS_UPPER  = [99.5, 98.6]


# ── Utility: read metrics CSV ─────────────────────────────────────────────────

def read_csv(path):
    """Read a metrics CSV and return a dict of lists keyed by column name."""
    if not os.path.isfile(path):
        return None
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return data if data else None


# ── Plot 1: Target accuracy comparison — Colored MNIST ───────────────────────

def plot_colored_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(COLORED_RESULTS["labels"]))
    bars = ax.bar(x, COLORED_RESULTS["target"], color=COLORED_RESULTS["colors"],
                  width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

    # Baseline reference line
    ax.axhline(COLORED_RESULTS["target"][0], color=COLORS["baseline"],
               linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)

    # Value labels
    for bar, val, delta in zip(bars, COLORED_RESULTS["target"], COLORED_RESULTS["delta"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        if delta != 0:
            sign = "+" if delta > 0 else ""
            color = "#22C55E" if delta > 0 else "#EF4444"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1.5,
                    f"{sign}{delta:.2f} pp", ha="center", va="top",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(COLORED_RESULTS["labels"], fontsize=11)
    ax.set_ylim(80, 101)
    ax.set_ylabel("Target Accuracy (%)")
    ax.set_title("Colored-MNIST (Binary) — Target Accuracy by Method")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/01_colored_mnist_comparison.png")
    plt.close(fig)
    print("Saved: 01_colored_mnist_comparison.png")


# ── Plot 2: Target accuracy comparison — Rotated MNIST ───────────────────────

def plot_rotated_comparison():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ROTATED_RESULTS["labels"]))
    bars = ax.bar(x, ROTATED_RESULTS["target"], color=ROTATED_RESULTS["colors"],
                  width=0.55, edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(ROTATED_RESULTS["target"][0], color=COLORS["baseline"],
               linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)

    for bar, val, delta in zip(bars, ROTATED_RESULTS["target"], ROTATED_RESULTS["delta"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
        if delta != 0:
            sign = "+" if delta > 0 else ""
            color = "#22C55E" if delta > 0 else "#EF4444"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 3.0,
                    f"{sign}{delta:.2f} pp", ha="center", va="top",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ROTATED_RESULTS["labels"], fontsize=11)
    ax.set_ylim(55, 105)
    ax.set_ylabel("Target Accuracy (%)")
    ax.set_title("Rotated-MNIST (10-class) — Target Accuracy by Method")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/02_rotated_mnist_comparison.png")
    plt.close(fig)
    print("Saved: 02_rotated_mnist_comparison.png")


# ── Plot 3: Side-by-side dataset comparison ───────────────────────────────────

def plot_side_by_side():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    method_labels = ["Source-only", "CORAL λ=1", "CORAL λ=10", "CORAL λ=25", "Upper bound"]
    x = np.arange(len(method_labels))
    w = 0.45

    # Colored
    ax1.bar(x, COLORED_RESULTS["target"], width=w, color=COLORED_RESULTS["colors"],
            edgecolor="white", zorder=3)
    ax1.axhline(COLORED_RESULTS["target"][0], color=COLORS["baseline"],
                linestyle="--", linewidth=1.2, alpha=0.5)
    ax1.set_ylim(80, 102)
    ax1.set_xticks(x); ax1.set_xticklabels(method_labels, fontsize=10, rotation=15, ha="right")
    ax1.set_ylabel("Target Accuracy (%)"); ax1.set_title("Colored-MNIST (Binary)\nStyle / texture shift")
    for bar, val in zip(ax1.patches, COLORED_RESULTS["target"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Rotated
    ax2.bar(x, ROTATED_RESULTS["target"], width=w, color=ROTATED_RESULTS["colors"],
            edgecolor="white", zorder=3)
    ax2.axhline(ROTATED_RESULTS["target"][0], color=COLORS["baseline"],
                linestyle="--", linewidth=1.2, alpha=0.5)
    ax2.set_ylim(55, 105)
    ax2.set_xticks(x); ax2.set_xticklabels(method_labels, fontsize=10, rotation=15, ha="right")
    ax2.set_ylabel("Target Accuracy (%)"); ax2.set_title("Rotated-MNIST (10-class)\nGeometric shift")
    for bar, val in zip(ax2.patches, ROTATED_RESULTS["target"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=COLORS["baseline"], label="Source-only baseline"),
        mpatches.Patch(color=COLORS["coral_l1"],  label="CORAL (positive)"),
        mpatches.Patch(color=COLORS["target"],    label="CORAL (negative transfer)"),
        mpatches.Patch(color=COLORS["upper"],     label="Upper bound"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)

    fig.suptitle("CORAL Adaptation: Style Shift vs Geometric Shift", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/03_side_by_side_comparison.png")
    plt.close(fig)
    print("Saved: 03_side_by_side_comparison.png")


# ── Plot 4: Lambda sensitivity ────────────────────────────────────────────────

def plot_lambda_sensitivity():
    lambdas = [1.0, 10.0, 25.0]

    colored_acc = [88.45, 89.91, 88.52]
    rotated_acc = [70.22, 68.03, 70.31]
    colored_base = 89.71
    rotated_base = 65.52

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, acc, base, title, ylim in [
        (ax1, colored_acc, colored_base, "Colored-MNIST (Binary)", (87, 91.5)),
        (ax2, rotated_acc, rotated_base, "Rotated-MNIST (10-class)", (63, 73)),
    ]:
        point_colors = ["#22C55E" if a > base else "#EF4444" for a in acc]
        ax.plot([str(l) for l in lambdas], acc, "o-", color="#0D9488",
                linewidth=2, markersize=9, zorder=3)
        for l, a, c in zip(lambdas, acc, point_colors):
            ax.plot(str(l), a, "o", color=c, markersize=11, zorder=4)
            ax.annotate(f"{a:.2f}%", (str(l), a),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=10, fontweight="bold")

        ax.axhline(base, color=COLORS["baseline"], linestyle="--",
                   linewidth=1.5, label=f"Baseline: {base:.2f}%", zorder=2)
        ax.set_ylim(ylim)
        ax.set_xlabel("Lambda (λ)"); ax.set_ylabel("Target Accuracy (%)")
        ax.set_title(title); ax.legend(fontsize=10)

    fig.suptitle("CORAL Loss Weight (λ) Sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/04_lambda_sensitivity.png")
    plt.close(fig)
    print("Saved: 04_lambda_sensitivity.png")


# ── Plot 5: Per-class accuracy — Rotated MNIST ────────────────────────────────

def plot_perclass_rotated():
    classes = [str(i) for i in range(10)]
    x = np.arange(10)
    w = 0.26

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w, ROTATED_PERCLASS_BEFORE, width=w, color=COLORS["baseline"],
           label="Source-only baseline", edgecolor="white", zorder=3)
    ax.bar(x,     ROTATED_PERCLASS_CORAL,  width=w, color=COLORS["coral_l1"],
           label="CORAL λ=25 (best)", edgecolor="white", zorder=3)
    ax.bar(x + w, ROTATED_PERCLASS_UPPER,  width=w, color=COLORS["upper"],
           label="Upper bound (supervised)", edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Digit {c}" for c in classes], fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Per-class Accuracy (%)")
    ax.set_title("Rotated-MNIST — Per-class Accuracy: Baseline vs CORAL vs Upper Bound")
    ax.legend(fontsize=11)

    # Annotate worst classes
    worst = [(i, ROTATED_PERCLASS_BEFORE[i]) for i in range(10)]
    worst.sort(key=lambda t: t[1])
    for idx, val in worst[:3]:
        ax.annotate(f"⚠ {val:.0f}%", (x[idx] - w, val),
                    textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8.5, color="#EF4444")

    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/05_perclass_rotated.png")
    plt.close(fig)
    print("Saved: 05_perclass_rotated.png")


# ── Plot 6: Per-class accuracy — Colored MNIST ───────────────────────────────

def plot_perclass_colored():
    classes = ["Class 0\n(digits 0–4)", "Class 1\n(digits 5–9)"]
    x = np.arange(2)
    w = 0.26

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(x - w, COLORED_PERCLASS_BEFORE, width=w, color=COLORS["baseline"],
           label="Source-only baseline", edgecolor="white", zorder=3)
    ax.bar(x,     COLORED_PERCLASS_CORAL,  width=w, color=COLORS["coral_l1"],
           label="CORAL λ=10 (best)", edgecolor="white", zorder=3)
    ax.bar(x + w, COLORED_PERCLASS_UPPER,  width=w, color=COLORS["upper"],
           label="Upper bound (supervised)", edgecolor="white", zorder=3)

    for bars, vals in [(x - w, COLORED_PERCLASS_BEFORE),
                       (x,     COLORED_PERCLASS_CORAL),
                       (x + w, COLORED_PERCLASS_UPPER)]:
        for xi, v in zip(bars, vals):
            ax.text(xi, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(80, 103)
    ax.set_ylabel("Per-class Accuracy (%)")
    ax.set_title("Colored-MNIST (Binary) — Per-class Accuracy")
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/06_perclass_colored.png")
    plt.close(fig)
    print("Saved: 06_perclass_colored.png")


# ── Plot 7: Training curves (from CSV if available) ───────────────────────────

def plot_training_curves(csv_path, title, output_name, dataset="colored"):
    data = read_csv(csv_path)
    if data is None:
        print(f"  Skipping {output_name} — {csv_path} not found")
        return

    epochs = list(range(1, len(data.get("epoch", data.get("val_acc", []))) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy curves
    ax = axes[0]
    if "val_acc" in data:
        ax.plot(epochs, [v * 100 for v in data["val_acc"]], color=COLORS["val"],
                linewidth=2, label="Val acc (source)")
    if "tgt_test_acc" in data:
        tgt = [v * 100 if isinstance(v, float) and v <= 1.0 else v
               for v in data["tgt_test_acc"]]
        ax.plot(epochs, tgt, color=COLORS["target"], linewidth=2,
                linestyle="--", label="Target test acc")
    if "train_acc" in data:
        ax.plot(epochs, [v * 100 for v in data["train_acc"]], color=COLORS["source"],
                linewidth=1.5, alpha=0.6, label="Train acc (source)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title}\nAccuracy Curves")
    ax.legend(fontsize=10)

    # Right: loss curves
    ax = axes[1]
    if "train_loss" in data:
        ax.plot(epochs, data["train_loss"], color=COLORS["source"],
                linewidth=2, label="Train loss")
    if "val_loss" in data:
        ax.plot(epochs, data["val_loss"], color=COLORS["val"],
                linewidth=2, label="Val loss")
    if "ce_loss" in data:
        ax.plot(epochs, data["ce_loss"], color=COLORS["ce"],
                linewidth=2, label="CE loss")
    if "coral_loss" in data:
        ax.plot(epochs, data["coral_loss"], color=COLORS["target"],
                linewidth=2, linestyle="--", label="CORAL loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{title}\nLoss Curves")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/{output_name}")
    plt.close(fig)
    print(f"Saved: {output_name}")


# ── Plot 8: Gap closed summary ────────────────────────────────────────────────

def plot_gap_closed():
    fig, ax = plt.subplots(figsize=(9, 5))

    datasets   = ["Colored-MNIST\n(Binary)", "Rotated-MNIST\n(10-class)"]
    total_gaps = [99.07 - 89.71, 99.19 - 65.52]   # upper - baseline
    coral_best = [89.91 - 89.71, 70.31 - 65.52]    # best CORAL - baseline
    pct_closed = [c / g * 100 for c, g in zip(coral_best, total_gaps)]

    x = np.arange(2)
    w = 0.35

    b1 = ax.bar(x - w/2, total_gaps, width=w, color=COLORS["upper"],
                alpha=0.35, edgecolor=COLORS["upper"], linewidth=1.5,
                label="Total available gap (upper bound − baseline)")
    b2 = ax.bar(x + w/2, coral_best, width=w, color=COLORS["coral_l1"],
                edgecolor="white", label="Gap closed by best CORAL")

    for i, (tot, cor, pct) in enumerate(zip(total_gaps, coral_best, pct_closed)):
        ax.text(i - w/2, tot + 0.2, f"{tot:.2f} pp", ha="center",
                fontsize=10, fontweight="bold", color=COLORS["upper"])
        ax.text(i + w/2, cor + 0.2, f"{cor:.2f} pp\n({pct:.0f}%)",
                ha="center", fontsize=10, fontweight="bold", color=COLORS["coral_l1"])

    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Accuracy improvement (pp)")
    ax.set_title("How Much of the Domain Gap Did CORAL Close?")
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/08_gap_closed.png")
    plt.close(fig)
    print("Saved: 08_gap_closed.png")


# ── Plot 9: Source vs target accuracy scatter ─────────────────────────────────

def plot_source_target_scatter():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, res, title in [
        (ax1, COLORED_RESULTS, "Colored-MNIST (Binary)"),
        (ax2, ROTATED_RESULTS, "Rotated-MNIST (10-class)"),
    ]:
        labels  = ["Source-only", "CORAL λ=1", "CORAL λ=10", "CORAL λ=25", "Upper bound"]
        markers = ["o", "s", "s", "s", "^"]
        sizes   = [100, 80, 80, 80, 120]

        for i, (src, tgt, col, lab, mk, sz) in enumerate(zip(
                res["source"], res["target"], res["colors"],
                labels, markers, sizes)):
            ax.scatter(src, tgt, color=col, marker=mk, s=sz, zorder=4,
                       edgecolors="white", linewidth=0.8)
            ax.annotate(lab, (src, tgt), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)

        # ideal line
        lims = [min(res["source"] + res["target"]) - 2,
                max(res["source"] + res["target"]) + 2]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.3, label="Source = Target (ideal)")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Source Accuracy (%)"); ax.set_ylabel("Target Accuracy (%)")
        ax.set_title(title); ax.legend(fontsize=9)

    fig.suptitle("Source vs Target Accuracy — All Methods", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/09_source_vs_target_scatter.png")
    plt.close(fig)
    print("Saved: 09_source_vs_target_scatter.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating plots...\n")

    # Summary / final-result plots (always generated from hardcoded data)
    plot_colored_comparison()
    plot_rotated_comparison()
    plot_side_by_side()
    plot_lambda_sensitivity()
    plot_perclass_rotated()
    plot_perclass_colored()
    plot_gap_closed()
    plot_source_target_scatter()

    # Training curve plots (generated only if CSV files exist)
    print("\nTraining curves (requires metrics.csv files):")
    plot_training_curves(
        "./results/source/metrics.csv",
        "Source Baseline — Colored-MNIST",
        "07a_curves_source_colored.png",
    )
    plot_training_curves(
        "./results/source_rot/metrics.csv",
        "Source Baseline — Rotated-MNIST",
        "07b_curves_source_rotated.png",
    )
    plot_training_curves(
        "./outputs/coral_rot/metrics.csv",
        "CORAL λ=1.0 — Rotated-MNIST",
        "07c_curves_coral_rot_l1.png",
    )
    plot_training_curves(
        "./outputs/coral_rotl10/metrics.csv",
        "CORAL λ=10.0 — Rotated-MNIST",
        "07d_curves_coral_rot_l10.png",
    )
    plot_training_curves(
        "./outputs/coral_rotl25/metrics.csv",
        "CORAL λ=25.0 — Rotated-MNIST",
        "07e_curves_coral_rot_l25.png",
    )
    plot_training_curves(
        "./outputs/coral_rot_l1/metrics.csv",
        "CORAL λ=1.0 — Colored-MNIST",
        "07f_curves_coral_colored_l1.png",
    )
    plot_training_curves(
        "./outputs/upper_bound_rot/metrics.csv",
        "Upper Bound — Rotated-MNIST",
        "07g_curves_upper_bound_rot.png",
    )

    print(f"\nAll plots saved to {PLOTS_DIR}/")