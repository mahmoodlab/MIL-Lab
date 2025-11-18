# MIL Cross-Validation Experiment Runner

This guide explains how to use `run_mil_experiments_cv.py` for k-fold cross-validation experiments.

## Overview

The CV experiment runner provides:
- **K-Fold Cross-Validation** - More robust evaluation than single hold-out
- **Stratified Splits** - Maintains class distribution across folds
- **Mean ± Std Metrics** - Statistical summary across all folds
- **Per-Fold Results** - Detailed metrics and confusion matrices for each fold
- **Multiple Experiments** - Run different MIL models and encoders with CV

## Key Differences from `run_mil_experiments.py`

| Feature | `run_mil_experiments.py` | `run_mil_experiments_cv.py` |
|---------|-------------------------|----------------------------|
| Evaluation | Single train/val/test split | K-fold cross-validation |
| Test Set | 10% hold-out | 20% per fold (rotating) |
| Metrics | Single value per experiment | Mean ± Std across folds |
| Models Trained | 1 per experiment | K per experiment (5 default) |
| Runtime | Faster | ~5x longer (for 5-fold) |
| Statistical Power | Lower | Higher |

## Quick Start

### 1. Configure Experiments

Edit `run_mil_experiments_cv.py`:

```python
EXPERIMENTS = [
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('transmil.base.uni_v2.pc108-24k', 'TransMIL + UNI_v2'),
]

FEATURE_PATHS = {
    'uni_v2': '/path/to/features_uni_v2/',
}

# Number of CV folds (default: 5)
N_FOLDS = 5
```

### 2. Run Experiments

```bash
python run_mil_experiments_cv.py
```

## How Cross-Validation Works

### K-Fold Strategy (Default: 5-Fold)

For each experiment, the data is split into 5 folds:

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

- Each fold uses ~80% for training and ~20% for testing
- Within each training set, 10% is held out for validation (early stopping)
- **Result**: Each data point is tested exactly once
- **Final metrics**: Mean ± Std across all 5 test sets

### Stratification

All splits maintain the original class distribution:
- If you have 30% class A, 50% class B, 20% class C
- Each fold will have approximately the same proportions

## Output Structure

```
experiment_results_cv/
└── run_20250117_143022/
    ├── experiment_config.json
    ├── cv_summary_results.csv              # Mean ± Std metrics
    ├── cv_detailed_results.csv             # Per-fold metrics
    │
    ├── best_model_abmil_uni_v2_fold1.pth
    ├── confusion_matrix_abmil_uni_v2_fold1.png
    ├── best_model_abmil_uni_v2_fold2.pth
    ├── confusion_matrix_abmil_uni_v2_fold2.png
    ├── ...
    └── confusion_matrix_abmil_uni_v2_fold5.png
```

## CSV Output Files

### 1. `cv_summary_results.csv`

**Aggregated statistics across all folds:**

| experiment_name | test_accuracy_mean | test_accuracy_std | test_balanced_accuracy_mean | ... |
|----------------|-------------------|------------------|---------------------------|-----|
| ABMIL + UNI_v2 | 0.8234 | 0.0156 | 0.7891 | ... |
| TransMIL + UNI_v2 | 0.8456 | 0.0123 | 0.8102 | ... |

**Columns:**
- `experiment_name` - Display name
- `model_config` - Model configuration string
- `n_folds` - Number of folds used
- `test_accuracy_mean` - Mean test accuracy
- `test_accuracy_std` - Standard deviation of test accuracy
- `test_balanced_accuracy_mean` - Mean balanced accuracy
- `test_balanced_accuracy_std` - Std of balanced accuracy
- `test_quadratic_kappa_mean` - Mean Cohen's Kappa
- `test_quadratic_kappa_std` - Std of Cohen's Kappa
- `best_val_loss_mean` - Mean best validation loss
- `best_val_loss_std` - Std of best validation loss

### 2. `cv_detailed_results.csv`

**Per-fold breakdown:**

| experiment_name | fold | test_accuracy | test_balanced_accuracy | test_quadratic_kappa | ... |
|----------------|------|---------------|----------------------|---------------------|-----|
| ABMIL + UNI_v2 | 1 | 0.8123 | 0.7823 | 0.8456 | ... |
| ABMIL + UNI_v2 | 2 | 0.8345 | 0.7967 | 0.8601 | ... |
| ... | ... | ... | ... | ... | ... |

**Columns:**
- `experiment_name` - Display name
- `model_config` - Model configuration
- `fold` - Fold number (1 to N_FOLDS)
- `test_accuracy` - Test accuracy for this fold
- `test_balanced_accuracy` - Balanced accuracy for this fold
- `test_quadratic_kappa` - Quadratic Kappa for this fold
- `best_val_loss` - Best validation loss
- `final_epoch` - Final training epoch
- `model_path` - Path to saved model
- `confusion_matrix_path` - Path to confusion matrix

## Configuration Options

### Cross-Validation Settings

```python
N_FOLDS = 5           # Number of folds (3, 5, or 10 common)
SEED = 10             # Random seed for fold splits
TRAIN_SEED = 42       # Seed for training (different per fold)
```

### Training Settings

Same as `run_mil_experiments.py`:

```python
NUM_EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS = 10
FEATURE_DROPOUT_RATE = 0.1
MODEL_DROPOUT_RATE = 0.25
```

## When to Use Cross-Validation

### ✅ Use CV When:
- You have **limited data** and need robust evaluation
- You want to **compare models statistically** (mean ± std)
- You need to **report confidence intervals**
- You're doing **hyperparameter tuning** (compare across settings)
- Publication requires **statistical validation**

### ❌ Use Single Split When:
- You have **very large datasets** (>10,000 samples)
- **Computational time** is critical
- You just need a **quick baseline** comparison
- The dataset has **natural temporal splits**

## Example Use Cases

### Example 1: Compare MIL Architectures

```python
EXPERIMENTS = [
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('transmil.base.uni_v2.pc108-24k', 'TransMIL + UNI_v2'),
    ('clam.base.uni_v2.pc108-24k', 'CLAM + UNI_v2'),
    ('dsmil.base.uni_v2.pc108-24k', 'DSMIL + UNI_v2'),
]

N_FOLDS = 5
```

**Result**: You can say "ABMIL achieved 82.3% ± 1.5% accuracy" with statistical confidence.

### Example 2: Compare Encoders

```python
EXPERIMENTS = [
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('abmil.base.gigapath.pc108-24k', 'ABMIL + GigaPath'),
    ('abmil.base.virchow2.pc108-24k', 'ABMIL + Virchow2'),
]

N_FOLDS = 5
```

**Result**: Determine which encoder performs best with confidence intervals.

### Example 3: 10-Fold for Small Dataset

If your dataset is small (<1000 samples):

```python
N_FOLDS = 10  # More folds for smaller datasets
```

## Interpreting Results

### Reading Summary Statistics

From `cv_summary_results.csv`:

```
test_accuracy_mean: 0.8234
test_accuracy_std: 0.0156
```

**Interpretation:**
- The model achieved **82.34% ± 1.56%** accuracy across 5 folds
- **95% confidence interval**: ~79.3% to 85.4% (mean ± 2*std)
- **Lower std = more stable** performance across folds

### Comparing Models

| Model | Accuracy | Std |
|-------|----------|-----|
| ABMIL | 82.34% | 1.56% |
| TransMIL | 84.56% | 1.23% |

**Conclusion**: TransMIL is likely better (higher mean, lower variance).

If confidence intervals overlap significantly, the difference may not be statistically significant.

## Tips & Best Practices

1. **Fold Count**:
   - Small datasets (<1000): Use 10-fold
   - Medium datasets (1000-10000): Use 5-fold
   - Large datasets (>10000): Use 3-fold or single split

2. **Random Seeds**:
   - `SEED` controls fold splitting (same across runs)
   - `TRAIN_SEED` varies per fold for model initialization

3. **Computational Cost**:
   - 5-fold CV trains 5 models per experiment
   - For 3 experiments: 15 total models
   - Plan accordingly for GPU time

4. **Early Stopping**:
   - Uses 10% of training data as validation
   - Prevents overfitting in each fold

5. **Ensemble Prediction**:
   - You can average predictions across all 5 fold models
   - Often improves performance (not implemented here)

## Statistical Significance Testing

To determine if one model is significantly better than another:

```python
from scipy import stats

# From cv_detailed_results.csv
abmil_scores = [0.812, 0.834, 0.818, 0.829, 0.824]  # Fold accuracies
transmil_scores = [0.845, 0.856, 0.841, 0.838, 0.848]

# Paired t-test (same folds)
t_stat, p_value = stats.ttest_rel(abmil_scores, transmil_scores)

if p_value < 0.05:
    print("TransMIL is significantly better than ABMIL (p < 0.05)")
```

## Runtime Estimates

Assuming 20 epochs, early stopping around epoch 15:

| Dataset Size | Single Split | 5-Fold CV | 10-Fold CV |
|-------------|--------------|-----------|------------|
| 1000 slides | ~30 min | ~2.5 hours | ~5 hours |
| 5000 slides | ~2 hours | ~10 hours | ~20 hours |
| 10000 slides | ~4 hours | ~20 hours | ~40 hours |

*Times are approximate and depend on GPU, batch size, and model complexity.*

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
- Reduce `BATCH_SIZE` (already 1)
- Use gradient checkpointing
- Run folds sequentially on different days

### Imbalanced Folds

If you see warnings about stratification:
- Check class distribution in your data
- Ensure you have enough samples per class
- Consider reducing `N_FOLDS` for very small classes

### Very High Variance

If std is very high (>5%):
- Dataset may be too small
- Consider more data augmentation
- Check for data quality issues
- Try different random seeds

## Available Models & Encoders

Same as `run_mil_experiments.py` - see `EXPERIMENT_RUNNER_README.md` for full lists.

**MIL Models**: `abmil`, `transmil`, `transformer`, `clam`, `dsmil`, `dftd`, `ilra`, `rrt`, `wikg`

**Encoders**: `uni_v2`, `gigapath`, `conch_v15`, `virchow2`, etc.

## Comparison to Original Script

| Script | Evaluation | Pros | Cons |
|--------|-----------|------|------|
| `run_mil.py` | Single run | Quick, simple | No variance estimate |
| `run_mil_experiments.py` | Single split | Fast, multiple experiments | No statistical validation |
| `run_mil_experiments_cv.py` | K-fold CV | Robust, mean±std | 5x slower |

Choose based on your needs!
