# Training Infrastructure Implementation - Session Notes

## Session 2: 2026-01-23 - Refactoring & Multi-Level MIL Support

### Completed Tasks

#### 1. Fixed PyTorch 2.6+ Compatibility
- Fixed `torch.load` issue with `weights_only=True` default
- Updated `trainer.py:load_checkpoint()` to use `weights_only=False`

#### 2. Created Shell Scripts for Training Pipeline
- `prepare_data.sh` - Data preparation script with configurable variables
- `train.sh` - Training script with configurable variables

#### 3. Added min_patches Filtering to `prepare_panda_data.py`
- Added `--min-patches` CLI argument (default: 24)
- Filters slides with insufficient patches

#### 4. Added CLI Args to `train_mil.py`
- `--early-stopping-patience` (default: 100)
- `--min-epochs` (default: 10)

#### 5. Updated `TrainConfig` Defaults
- `early_stopping_patience`: 10 → 100
- `min_epochs`: 5 → 10

#### 6. Refactored `data_loading/dataset.py` for Multi-Level MIL
Major refactor to support different aggregation levels:

**New Architecture:**
```
MILDataset (slide-level)
    │
    ├── .concat_by('column') → GroupedMILDataset (flat concatenation)
    │
    └── .group_by('column')  → HierarchicalMILDataset (preserves structure)
```

**New Classes:**
- `GroupedData` - Container for grouped/concatenated data
- `HierarchicalData` - Container for hierarchical data (list of tensors)
- `GroupedMILDataset` - Flat concatenation of items in a group
- `HierarchicalMILDataset` - Preserves item structure for two-stage attention

**Backwards Compatibility:**
- `CaseData` = alias for `GroupedData`
- `CaseMILDataset` = alias for `GroupedMILDataset`

### Design Decisions

#### Multi-Level MIL Support
The refactor addresses three use cases:

1. **Slide-level** (`MILDataset`):
   - One bag per H5 file
   - Use for: PANDA (1 patient = 1 slide)

2. **Grouped/Flat** (`GroupedMILDataset` via `.concat_by()`):
   - Concatenates multiple slides into one bag
   - Use for: Multi-slide patients where slide boundaries don't matter

3. **Hierarchical** (`HierarchicalMILDataset` via `.group_by()`):
   - Preserves slide structure within patient
   - Use for: Two-stage attention (patch→slide→patient)

#### Why This Design?
- Single `MILDataset` class with methods for grouping
- No duplication of code for different levels
- Chainable: `dataset.concat_by('slide_id').concat_by('case_id')` theoretically possible
- Clear separation: data loading provides data, model decides aggregation strategy

---

## Session 1: 2026-01-22 - Initial Implementation

### Completed Tasks

#### 1. Created `data_loading/pytorch_adapter.py`
Bridge between MILDataset and PyTorch DataLoader:
- `MILDatasetAdapter` - Wraps MILDataset as torch Dataset with label encoding
- `mil_collate_fn` - Handles variable-length bags with padding/masks
- `create_dataloader()` - Factory with weighted sampling support

#### 2. Created `training/config.py`
Dataclass configurations:
- `DataConfig` - labels_csv, features_dir, split settings
- `TrainConfig` - epochs, lr, dropout, early stopping params
- `ExperimentConfig` - Top-level config with JSON save/load

#### 3. Created `training/trainer.py`
`MILTrainer` class with:
- AMP (mixed precision) training
- Gradient clipping
- Early stopping based on validation kappa
- Best model checkpointing
- History dict tracking (MLflow-ready)

#### 4. Created `training/evaluator.py`
- `evaluate()` - Full model evaluation
- `compute_metrics()` - accuracy, balanced_acc, kappa, confusion_matrix
- `print_evaluation_results()` - Formatted output

#### 5. Created `training/__init__.py`
Public API exports for the training module.

#### 6. Created `train_mil.py`
Entry point script that:
- Loads data using `MILDataset`
- Creates model using `src.builder.create_model()`
- Trains with `MILTrainer`
- Evaluates and saves results

#### 7. Created `prepare_panda_data.py`
Utility to convert PathBench TSV format to labels CSV format expected by MILDataset.

---

## File Structure

```
MIL-Lab/
├── data_loading/
│   ├── __init__.py          # Updated exports
│   ├── dataset.py           # MILDataset, GroupedMILDataset, HierarchicalMILDataset
│   ├── feature_loader.py    # H5 feature loading
│   └── pytorch_adapter.py   # PyTorch DataLoader integration
├── training/
│   ├── __init__.py          # Public API
│   ├── config.py            # DataConfig, TrainConfig, ExperimentConfig
│   ├── trainer.py           # MILTrainer
│   ├── evaluator.py         # evaluate(), compute_metrics()
│   └── SESSION_NOTES.md     # This file
├── train_mil.py             # Entry point
├── prepare_panda_data.py    # Data preparation utility
├── prepare_data.sh          # Data prep shell script
└── train.sh                 # Training shell script
```

---

## Usage

### For PANDA (1 slide = 1 patient)
```bash
# Prepare data
./prepare_data.sh

# Train
./train.sh
```

### For Multi-Slide Datasets
```python
from data_loading import MILDataset

# Load at slide level
dataset = MILDataset('labels.csv', 'features/')

# Group by patient (flat concat)
patient_dataset = dataset.concat_by('case_id')

# Or hierarchical (for two-stage MIL)
patient_dataset = dataset.group_by('case_id')

# Split and train
splits = patient_dataset.random_split(stratify=True)
```

---

## Original Script Settings (for replication)

From `run_mil_experiments_predefined_splits.py`:
- `NUM_EPOCHS = 20`
- `BATCH_SIZE = 1`
- `LEARNING_RATE = 1e-4`
- `WEIGHT_DECAY = 1e-5`
- `EARLY_STOPPING_PATIENCE = 100`
- `MIN_EPOCHS = 10`
- `FEATURE_DROPOUT_RATE = 0.1`
- `MODEL_DROPOUT_RATE = 0.25`
- `SEED = 10` (data split)
- `TRAIN_SEED = 42` (training)
- Model: `'abmil.base.uni_v2.pc108-24k'`

All these are now the defaults in `TrainConfig`.
