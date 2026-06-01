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

## Methods

1. Source-only CNN
   - trains only on labeled source data
   - target data is used only for evaluation

2. Deep CORAL
   - starts from the source-only checkpoint
   - uses source images + source labels
   - uses target images without target labels
   - loss = classification loss + lambda * CORAL loss

3. Target-supervised upper bound
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
- we keep the best validation model, not just the last epoch
- long cluster jobs can be resumed or inspected

### Do not tune on the final test set

Do not choose hyperparameters secretly based on final target test accuracy.

If you run many lambdas and show all results, call it an ablation/sensitivity study.
Do not hide failed values.

## Recommended local commands

Train source-only baseline:

```bash
python src/train_source.py --epochs 5 --target_correlation 0.10
```

Train Deep CORAL:

```bash
python src/train_coral.py --epochs 5 --lambda_coral 10 --target_correlation 0.10
```

Train upper bound:

```bash
python src/train_upper_bound.py --epochs 5 --target_correlation 0.10
```

Run a smaller CPU-friendly sweep:

```bash
python src/sweep_lambdas.py --epochs 3 --lambdas 0.1 1 5 10 25 --target_correlations 0.10 0.30 0.50
```

Plot heatmap:

```bash
python src/plot_sweep_heatmap.py --value delta
```

## Expected outputs

Results are saved under:

```text
results/
```

Checkpoints are saved under:

```text
checkpoints/
```

Do not push these folders to GitHub unless they are small final result summaries.
