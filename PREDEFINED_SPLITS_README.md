# Using Predefined Splits for Training

This guide explains how to use predefined train/test splits from `k=all.tsv` instead of random splitting.

## Overview

The PANDA dataset comes with a predefined split file (`k=all.tsv`) that specifies which slides should be used for training and testing. This ensures reproducibility and allows for fair comparison with other published work.

## Key Differences

### Random Splits (Original)
- **Script**: `run_mil_experiments.py`, `run_mil_experiments_cv.py`, `run_mil_experiments_gem3.py`
- **Function**: `preprocess_panda_data()`
- **Input**: `train.csv` (10,616 slides)
- **Split**: Random 80/10/10 split using `train_test_split()`
- **Use Case**: Quick experiments, when you want different random splits

### Predefined Splits (New)
- **Script**: `run_mil_experiments_predefined_splits.py`
- **Function**: `load_panda_predefined_splits()`
- **Input**: `k=all.tsv` (9,555 slides with predefined train/test)
- **Split**: Uses existing train/test split from TSV + optional validation split
- **Use Case**: Reproducible experiments, fair comparison with literature

## Data Statistics

### k=all.tsv Split Distribution
- **Total slides**: 9,555 (subset of train.csv)
- **Training**: 8,601 slides (90%)
- **Test**: 954 slides (10%)
- **Validation**: Created from training set (configurable)

### After Feature Matching & QC (≥24 patches)
- **Total slides**: 9,520
- **Training**: 7,711 slides (81.0%)
- **Validation**: 857 slides (9.0%) - if `VAL_FRACTION=0.1`
- **Test**: 952 slides (10.0%)

## Configuration

Edit the configuration section in `run_mil_experiments_predefined_splits.py`:

```python
# Data paths
TSV_PATH = '/path/to/k=all.tsv'  # Predefined splits

# Data settings
FOLD_COLUMN = 'fold_0'      # Column in TSV containing train/test split
VAL_FRACTION = 0.1           # Create 10% validation from training (0 = no validation)
GRADE_GROUP = False          # Use original ISUP grades (0-5)
EXCLUDE_MID_GRADE = False    # Only applies if GRADE_GROUP=True
SEED = 10                    # Seed for validation split
TRAIN_SEED = 42              # Seed for model initialization & sampling
```

## Usage

### Run Experiments with Predefined Splits

```bash
python run_mil_experiments_predefined_splits.py
```

### Test the Functionality

```bash
python test_predefined_splits.py
```

## Function Signature

```python
load_panda_predefined_splits(
    tsv_path,           # Path to k=all.tsv
    feats_path,         # Path to feature directory
    fold_column='fold_0',      # Column with train/test labels
    val_fraction=0.1,          # Fraction of training data for validation
    grade_group=False,         # Whether to group ISUP grades
    exclude_mid_grade=False,   # Exclude ISUP 2-3 (if grade_group=True)
    min_patches=24,            # Minimum patches per slide
    seed=10                    # Random seed for validation split
)
```

## Reproducibility

The predefined splits ensure:
1. ✅ Same train/test split as defined in `k=all.tsv`
2. ✅ Reproducible validation split (with `seed=10`)
3. ✅ Fair comparison with other work using the same splits
4. ✅ All labels match between `train.csv` and `k=all.tsv` (100% verified)

## Validation Split Strategy

The validation set is created by:
1. Taking the predefined **training set** from `k=all.tsv`
2. Randomly splitting it into train/val using stratified sampling
3. Keeping the **test set** unchanged from `k=all.tsv`

Example with `VAL_FRACTION=0.1`:
- Original train: 8,568 slides
- New train: ~7,711 slides (90% of original train)
- Validation: ~857 slides (10% of original train)
- Test: 952 slides (unchanged)

## Output

Results are saved to: `experiment_results_predefined/run_TIMESTAMP/`

Includes:
- `experiment_results.csv` - Test metrics for all experiments
- `confusion_matrix_*.png` - Confusion matrices
- `best_model_*.pth` - Saved model weights
- `experiment_config.json` - Full experiment configuration

## Notes

- The test set from `k=all.tsv` is **never used for validation** - it remains held out
- 21 slides from `k=all.tsv` are missing feature files (not extracted)
- 14 slides have <24 patches and are filtered out during QC
- All 9,555 slides in `k=all.tsv` have matching labels in `train.csv` (100% match rate)
