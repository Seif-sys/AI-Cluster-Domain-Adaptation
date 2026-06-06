# Project 4 — Domain Adaptation Upgrade

This upgraded version keeps the project simple but makes it scientifically stronger.

## Main experiment

Dataset: Binary Colored-MNIST

- class 0: digits 0, 1, 2, 3, 4
- class 1: digits 5, 6, 7, 8, 9

Source domain:

- strong color-label correlation, usually `0.99`

Target domain:

- weaker or different color-label correlation, for example `0.10`, `0.30`, or `0.50`

Example:

```text
source_correlation = 0.99
target_correlation = 0.10
```

This means the color rule is strong in the source domain but mostly broken/flipped in the target domain.

## Methods

1. Source-only CNN
   - trains only on labeled source data
   - target data is used only for evaluation
   - this is the baseline

2. Deep CORAL
   - starts from the source-only checkpoint
   - uses source images + source labels
   - uses target images without target labels
   - target labels are not used during adaptation
   - loss = classification loss + lambda * CORAL loss

3. Pseudo-labeling
   - starts from the source-only checkpoint
   - predicts labels for target images
   - keeps only confident predictions
   - trains using source labels + pseudo-target labels
   - does not use real target labels
   - confidence threshold controls how strict we are

4. Target-supervised upper bound
   - uses target labels
   - not a UDA method
   - only shows the best-case comparison

## Important terms

### Lambda

`lambda_coral` controls how strong the CORAL loss is.

Small lambda: model mostly focuses on source classification.  
Large lambda: model strongly tries to align source/target features.

### Validation split

A validation split is a small part of the training data used to choose the best checkpoint.

It is not the final test set.

We use validation accuracy to save the best model checkpoint.

### Checkpoint

A checkpoint is a saved model.

It is useful because:

- Deep CORAL starts from the source-only checkpoint
- pseudo-labeling starts from the source-only checkpoint
- we keep the best validation model, not just the last epoch
- long cluster jobs can be resumed or inspected

### Do not tune on the final test set

Do not choose hyperparameters secretly based on final target test accuracy.

If you run many lambdas and show all results, call it an ablation/sensitivity study.

Do not hide failed values.

## How to use the project with `run_project.py`

The file `run_project.py` is a helper script that runs project commands from one place.

Instead of manually typing every training and plotting command, we can select which part of the project should run.

General format:

```bash
python run_project.py --steps <step_name>
```

Available step names:

```text
source      train the source-only CNN
coral       train Deep CORAL
upper       train the target-supervised upper bound
pseudo      train pseudo-labeling
sweep       run lambda / target-correlation sweep
heatmap     create heatmap from sweep results
final       collect important final result files
plots       create presentation plots
pca         create PCA feature-space visualization

core        shortcut for: source + coral + upper + final
analysis    shortcut for: final + plots + pca
sweep_all   shortcut for: sweep + heatmap + final + plots
all         runs almost everything
```

Examples:

```bash
python run_project.py --steps core

python run_project.py --steps coral --lambda_coral 10 --target_correlation 0.10

python run_project.py --steps pseudo --confidence_threshold 0.95

python run_project.py --steps pca

python run_project.py --steps sweep_all

python run_project.py --steps all
```

Warning: `all` can take longer because it runs training, sweep, plotting, PCA, and pseudo-labeling.

## Recommended local commands

The project can be run either with `run_project.py` or manually.

### Train source-only baseline

```bash
python src/train_source.py --epochs 5 --target_correlation 0.10
```

### Train Deep CORAL

```bash
python src/train_coral.py --epochs 5 --lambda_coral 10 --target_correlation 0.10
```

### Train target-supervised upper bound

```bash
python src/train_upper_bound.py --epochs 5 --target_correlation 0.10
```

### Train pseudo-labeling

```bash
python src/train_pseudo_label.py --source_checkpoint checkpoints/source_only_best.pt --epochs 5 --target_correlation 0.10 --confidence_threshold 0.95
```

If no pseudo-labels are selected, lower the threshold:

```bash
python src/train_pseudo_label.py --source_checkpoint checkpoints/source_only_best.pt --epochs 5 --target_correlation 0.10 --confidence_threshold 0.90
```

### Run a smaller CPU-friendly sweep

```bash
python src/sweep_lambdas.py --epochs 3 --lambdas 0.1 1 5 10 25 --target_correlations 0.10 0.30 0.50
```

### Plot heatmap

```bash
python src/plot_sweep_heatmap.py --value delta
```

### Create presentation plots

```bash
python src/make_presentation_plots.py
```

### Create PCA feature visualization

```bash
python src/plot_pca_features.py --source_checkpoint checkpoints/source_only_best.pt --coral_checkpoint checkpoints/coral_best.pt --target_correlation 0.10
```

## Expected outputs

Training results are saved under:

```text
results/
results_cloud/
```

Checkpoints are saved under:

```text
checkpoints/
```

Final presentation files are collected under:

```text
results_cloud/final/
```

Final presentation plots are saved under:

```text
results_cloud/final/plots/
```

Important generated files include:

```text
source_only_results.json
deep_coral_results.json
upper_bound_results.json
sweep_results.csv
target_accuracy_bar.png
error_reduction.png
sweep_heatmap_clean.png
pca_source_only.png
pca_deep_coral.png
```

Do not push large generated folders to GitHub unless they are small final result summaries.

Usually ignored:

```text
data/
checkpoints/
results/
results_cloud/
*.pt
*.pth
*.log
*.out
```

## Git notes

After adding the runner and README updates:

```bash
git add README.md run_project.py
git commit -m "Add project runner and extended usage instructions"
git push
```