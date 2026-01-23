# Session Notes: Data Loading Module

**Last Updated:** 2026-01-21
**Status:** Core MILDataset complete, ready for DataLoader integration

---

## Summary

Built a clean data loading module for MIL-Lab that:
1. Loads CLAM-extracted H5 features
2. Loads labels from CSV
3. Joins them on `slide_id`
4. Supports both predefined splits (PANDA) and random splits (Enclave)

---

## Files

```
data_loading/
├── __init__.py                    # Public API
├── feature_loader.py              # H5 feature loading (renamed from clam_dataloader.py)
├── dataset.py                     # CSV labels + join logic + MILDataset
├── test_clam_compatibility.py     # Unit tests
├── SPECIFICATION.md               # Formal specification
└── SESSION_NOTES.md               # This file
```

---

## Key Design Decisions

1. **Separation of concerns**: Labels and features loaded independently, joined by slide_id
2. **Dagster-ready**: Functions take DataFrames (not paths) for composability
3. **Lazy loading**: Features loaded on access, not upfront
4. **String labels**: Both `slide_id` and `label` are strings
5. **Computed at init**: `embed_dim` and `num_classes` computed once (not @property)
6. **Dual split support**: `split_by_column()` for PANDA, `random_split()` for Enclave

---

## MILDataset API

```python
from data_loading import MILDataset

# Initialize
dataset = MILDataset(
    labels_csv='path/to/labels.csv',      # Required: slide_id, label columns
    features_dir='path/to/h5_files/',     # Directory with {slide_id}.h5 files
)

# Properties
len(dataset)          # Number of slides
dataset.embed_dim     # Embedding dimension (e.g., 1536)
dataset.num_classes   # Number of unique labels
dataset.slide_ids     # List of all slide IDs
dataset.labels        # List of all labels

# Access
slide = dataset[0]              # By index
slide = dataset['slide_001']    # By slide_id
slide.slide_id                  # 'slide_001'
slide.features                  # Tensor [M, D]
slide.label                     # 'cancer' (string)
slide.case_id                   # 'patient_A' or None

# Iterate
for slide in dataset:
    print(slide.slide_id, slide.features.shape)

# Split - Option 1: By column (PANDA with split column)
splits = dataset.split_by_column('split')
train_data = splits['train']
val_data = splits['val']
test_data = splits['test']

# Split - Option 2: Random (Enclave without split column)
splits = dataset.random_split(train_frac=0.7, val_frac=0.15, seed=42)
train_data = splits['train']
val_data = splits['val']
test_data = splits['test']
```

---

## CSV Format Expected

```csv
slide_id,label,case_id
slide_001,cancer,patient_A
slide_002,normal,patient_B
```

Required columns: `slide_id`, `label`
Optional columns: `case_id`, `split`, any others (preserved)

---

## Implementation Checklist

### Completed
- [x] `load_features()` - Load single H5 file
- [x] `get_embed_dim()` - Get embedding dimension
- [x] `get_slide_ids()` - List available slides
- [x] `SlideData` - Data container (dataclass)
- [x] `load_labels()` - Load CSV with string labels
- [x] `get_available_features()` - Scan H5 directory
- [x] `join_labels_and_features()` - Inner join (both DataFrames)
- [x] `MILDataset` class with:
  - [x] `__init__`, `__len__`, `__getitem__`, `__iter__`
  - [x] `get_subset()`
  - [x] `split_by_column()` - For PANDA data
  - [x] `random_split()` - For Enclave data
- [x] Renamed `clam_dataloader.py` → `feature_loader.py`
- [x] Removed test blocks from production code

### Next Steps (Priority Order)

#### Immediate (First Version)
1. [ ] Test in enclave with real data

#### HIGH Priority (Before Dagster/MLflow)
2. [ ] Add `validate_h5_files()` function - check embed_dim consistency, detect corrupt files
3. [ ] Move `create_dataloaders()` into `data_loading` module (from `utils/data_utils.py`)
4. [ ] Add `MILDataset.from_manifest()` classmethod - bridge between Dagster assets and PyTorch
5. [ ] Add configuration management (`data_loading/config.py` with Pydantic)

#### MEDIUM Priority (During Dagster/MLflow Integration)
6. [ ] Create Dagster asset definitions for data pipeline
7. [ ] Add MLflow logging to assets (data versioning, manifests)
8. [ ] Add data quality metrics (class balance entropy, patch distribution)
9. [ ] Add weighted sampling for class imbalance
10. [ ] Add stratified splitting (preserve label distribution)

#### LOW Priority (Production Hardening)
11. [ ] Error handling for corrupt H5 files
12. [ ] H5 file handle management for multi-worker DataLoader
13. [ ] Unit tests for splits, error handling, edge cases
14. [ ] Monitoring hooks (load times, error rates)

---

## Architecture Review Summary (MLOps)

**Grade: B+** (Design) / **C** (Production Readiness)
**Dagster/MLflow Readiness:** 70%

### Strengths
- Clean separation of concerns
- Dagster-ready design (pure functions, DataFrame-based)
- Lazy loading pattern
- Dual split support (predefined + random)

### Issues to Address
1. No H5 validation layer (embed_dim consistency, corrupt files)
2. No `from_manifest()` bridge for Dagster → PyTorch
3. Missing DataLoader integration in this module
4. No configuration management
5. No MLflow logging hooks

---

## Recommended Two-Layer Architecture (Dagster + PyTorch)

```
┌─────────────────────────────────────┐
│        DAGSTER ASSETS               │
│  (Pure functions → DataFrames)      │
│                                     │
│  load_labels() → validate_h5() →    │
│  join_data() → split_data() →       │
│  manifests (train/val/test DFs)     │
└────────────────┬────────────────────┘
                 │ (DataFrames persisted & versioned)
                 ↓
┌─────────────────────────────────────┐
│     PYTORCH DATALOADER              │
│  (Wraps manifests for training)     │
│                                     │
│  MILDataset.from_manifest(df) →     │
│  DataLoader → Training Loop         │
└─────────────────────────────────────┘
```

### Key Code to Add

**1. H5 Validation Function (feature_loader.py)**
```python
def validate_h5_files(features_df: pd.DataFrame) -> pd.DataFrame:
    """Validate H5 files, return DataFrame with num_patches, embed_dim, is_valid."""
    ...
```

**2. From Manifest Classmethod (dataset.py)**
```python
@classmethod
def from_manifest(cls, manifest_df: pd.DataFrame, features_dir: Path) -> 'MILDataset':
    """Create MILDataset from pre-validated manifest (Dagster bridge)."""
    dataset = cls.__new__(cls)
    dataset.features_dir = Path(features_dir)
    dataset.df = manifest_df.copy()
    dataset._index = {row.slide_id: idx for idx, row in dataset.df.iterrows()}
    dataset.embed_dim = int(manifest_df['embed_dim'].iloc[0])
    dataset.num_classes = manifest_df['label'].nunique()
    return dataset
```

**3. DataLoader Factory (dataloader.py - new file)**
```python
def create_mil_dataloaders(splits: Dict[str, MILDataset], ...) -> Tuple[DataLoader, ...]:
    """Create train/val/test DataLoaders with proper settings."""
    ...

def mil_collate_fn(batch: List[SlideData]) -> Dict[str, torch.Tensor]:
    """Collate for variable-length MIL bags with padding."""
    ...
```

### Dagster Asset Graph (Future)

```
panda_labels ─────┐
                  ├──► validated_dataset ──► dataset_with_splits
h5_file_inventory ┘         │                      │
        │                   ▼                      ├──► train_manifest
        ▼           validated_h5_metadata          ├──► val_manifest
   (scan dir)       (check embed_dim)              └──► test_manifest
                                                            │
                                                            ▼
                                                    training_job (MLflow)
```

### MLflow Logging Points

| Point | What to Log |
|-------|-------------|
| Data validation | num_slides, embed_dim, class_counts |
| Split creation | manifest CSVs as artifacts, split sizes |
| Training | hyperparams, epoch metrics, model checkpoint |
| Evaluation | test metrics, confusion matrix |

---

## Usage with Training Script

```python
from data_loading import MILDataset
import torch
from torch.utils.data import DataLoader

# Load dataset
dataset = MILDataset('labels.csv', 'features/')
splits = dataset.random_split(train_frac=0.7, val_frac=0.15, seed=42)

# Manual DataLoader (until create_dataloaders is added)
train_loader = DataLoader(splits['train'], batch_size=1, shuffle=True)
val_loader = DataLoader(splits['val'], batch_size=1, shuffle=False)
test_loader = DataLoader(splits['test'], batch_size=1, shuffle=False)

# Training loop
for slide in train_loader:
    features = slide.features  # [1, M, D]
    label = slide.label
    # ... training code
```

---

## Related Files

- `utils/data_utils.py` - Original MIL-Lab data loading (PANDAH5Dataset)
- `run_mil_experiments_predefined_splits.py` - Training script using current system
- `src/models/abmil.py` - ABMIL model (expects [B, M, D] input)

---

## Revision History

| Date | Changes |
|------|---------|
| 2025-01-15 | Initial module creation |
| 2026-01-21 | Added random_split(), string labels, renamed to feature_loader.py, Dagster-ready join function |
| 2026-01-22 | MLOps architecture review, added Dagster/MLflow integration plan, prioritized action items |
