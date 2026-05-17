# Project 4 — Domain Adaptation

## Main idea

We study domain adaptation. A model is trained on a labeled source domain and adapted using target-domain data. The goal is to improve performance on the target domain.

## Research question

Does Deep CORAL improve target-domain accuracy compared to a source-only CNN baseline?

## Dataset choice

We use Colored-MNIST binary classification.

Original MNIST digits are converted into two classes:

- class 0: digits 0, 1, 2, 3, 4
- class 1: digits 5, 6, 7, 8, 9

## Source domain

Source domain = Colored-MNIST with strong color-label correlation.

Example:

- class 0 digits are usually red
- class 1 digits are usually green
- 99% of samples use the correct color
- 1% of samples use the opposite color

Labels are available and used for training.

## Target domain

Target domain = Colored-MNIST with changed color-label correlation.

Example:

- color is random, or
- color-label correlation is much weaker than in the source domain

For unsupervised domain adaptation, target images are used during adaptation, but target labels are not used.

Target labels are used only for final evaluation.

## Models / methods

We compare three settings:

1. Source-only baseline  
   Train a CNN only on labeled source data.

2. Deep CORAL adaptation  
   Train using labeled source data and unlabeled target data.  
   The classification loss uses source labels.  
   The CORAL loss aligns source and target feature statistics.

3. Target-supervised upper bound  
   Train or fine-tune using target labels.  
   This is not unsupervised domain adaptation.  
   It is only an upper bound comparison.


Method	                        Source images	Source labels	Target images	           Target labels
Source-only CNN	                yes	        yes	        no training, only testing  only for evaluation
Deep CORAL	                yes	        yes	        yes	                   no
Target-supervised upper bound	maybe	        maybe	        yes	                   yes

## Metrics

We report:

- source test accuracy before adaptation
- target test accuracy before adaptation
- target test accuracy after Deep CORAL
- target-supervised upper bound accuracy
- per-class target accuracy
- confusion matrix
- whether adaptation helped or caused negative transfer

## Rules

Target test labels are never used to train or tune the adaptation method.

Random seed:

- seed = 42

We will report:

- hardware used
- training time
- number of training runs
- main hyperparameters



## Project structure

domain-adaptation-project/
  README.md
  requirements.txt

  data/
    Stores downloaded or generated datasets.

  src/
    data_colored_mnist.py
      Creates source and target Colored-MNIST datasets.

    models.py
      Contains the CNN model.

    train_source.py
      Trains the source-only CNN baseline.

    train_coral.py
      Trains the Deep CORAL domain adaptation model.

    train_upper_bound.py
      Trains the target-supervised upper bound.

    evaluate.py
      Evaluates accuracy, per-class accuracy, and confusion matrix.

    utils.py
      Contains helper functions such as seed setting, accuracy calculation, saving and loading models.

  scripts/
    run_source.sh
      Runs source-only training.

    run_coral.sh
      Runs Deep CORAL training.

    run_upper_bound.sh
      Runs target-supervised upper bound training.

  results/
    Stores result files, plots, confusion matrices, and logs.

  checkpoints/
    Stores trained model checkpoints.