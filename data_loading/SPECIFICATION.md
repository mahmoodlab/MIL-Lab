# Data Loading Module Specification

**Version:** 1.0
**Date:** 2025-01-15
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose

This module provides a standardized interface for loading and preparing data for Multiple Instance Learning (MIL) models. It handles two distinct data sources and joins them into a unified dataset.

### 1.2 Scope

- Loading slide-level labels from CSV files
- Loading patch-level features from HDF5 files (CLAM format)
- Joining labels and features by slide identifier
- Preparing tensors for MIL model input

### 1.3 Out of Scope

- Feature extraction from whole slide images
- Model training and evaluation
- Data augmentation
- Distributed data loading

---

## 2. Data Sources

### 2.1 Labels Source

**Format:** CSV file

**Required Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `slide_id` | string | Unique slide identifier (must match H5 filename) |
| `label` | integer | Class label (0-indexed) |

**Optional Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `case_id` | string | Patient/case identifier (for patient-level splits) |
| `split` | string | Dataset split assignment (`train`, `val`, `test`) |

**Example:**

```csv
slide_id,case_id,label,split
slide_001,patient_A,0,train
slide_002,patient_A,0,train
slide_003,patient_B,1,val
slide_004,patient_C,2,test
```

### 2.2 Features Source

**Format:** Directory of HDF5 files

**Naming Convention:** `{slide_id}.h5`

**HDF5 Schema:**

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `features` | `(M, D)` | `float32` | Patch embeddings |
| `coords` | `(M, 2)` | `int32` | Patch coordinates (optional, not loaded) |

Where:
- `M` = number of patches (variable per slide)
- `D` = embedding dimension (constant across slides)

**Supported Embedding Dimensions:**

| Encoder | Dimension |
|---------|-----------|
| ResNet50 | 1024 |
| UNI v1 | 1024 |
| UNI v2 | 1536 |
| CONCH v1 | 512 |
| CONCH v1.5 | 768 |
| GigaPath | 1536 |

**Alternative Format (Legacy):**

Shape `(1, M, D)` is also accepted and automatically squeezed to `(M, D)`.

---

## 3. Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Layer 3: Interface                    │
│                                                             │
│                         MILDataset                          │
│                             │                               │
│              ┌──────────────┴──────────────┐                │
│              ▼                             ▼                │
│     ┌─────────────────┐          ┌─────────────────┐        │
│     │  Labels (CSV)   │          │ Features (H5)   │        │
│     └────────┬────────┘          └────────┬────────┘        │
│              │                            │                 │
├──────────────┼────────────────────────────┼─────────────────┤
│              │        Layer 2: Join       │                 │
│              │                            │                 │
│              └──────────┬─────────────────┘                 │
│                         │                                   │
│            join_labels_and_features()                       │
│                         │                                   │
│                         ▼                                   │
│                   matched_df                                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                     Layer 1: Raw I/O                        │
│                                                             │
│     load_labels()                    load_features()        │
│     get_available_features()         get_embed_dim()        │
│                                      get_slide_ids()        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
data_loading/
├── __init__.py              # Public API exports
├── clam_dataloader.py       # Layer 1: H5 feature loading
├── dataset.py               # Layer 2-3: CSV + join + MILDataset
├── test_clam_compatibility.py
└── SPECIFICATION.md         # This document
```

---

## 4. Public API

### 4.1 Layer 1: Raw I/O

#### `load_features(h5_path) → Tensor`

Load features from a single HDF5 file.

**Parameters:**
- `h5_path`: Path to `.h5` file

**Returns:**
- `Tensor` of shape `[M, D]`

**Raises:**
- `FileNotFoundError` if file does not exist
- `KeyError` if `features` dataset not in file

---

#### `get_slide_ids(features_dir) → List[str]`

Get all slide IDs from a features directory.

**Parameters:**
- `features_dir`: Directory containing `.h5` files

**Returns:**
- Sorted list of slide IDs (filenames without `.h5` extension)

---

#### `get_embed_dim(features_dir) → int`

Infer embedding dimension from feature files.

**Parameters:**
- `features_dir`: Directory containing `.h5` files

**Returns:**
- Integer embedding dimension

**Raises:**
- `FileNotFoundError` if no `.h5` files in directory

---

#### `load_labels(csv_path) → DataFrame`

Load labels from CSV file.

**Parameters:**
- `csv_path`: Path to CSV file

**Returns:**
- DataFrame with at least `slide_id`, `label` columns

**Raises:**
- `ValueError` if required columns missing

---

### 4.2 Layer 2: Join

#### `get_available_features(features_dir) → DataFrame`

Get metadata about available feature files.

**Parameters:**
- `features_dir`: Directory containing `.h5` files

**Returns:**
- DataFrame with columns: `slide_id`, `h5_path`

---

#### `join_labels_and_features(labels_df, features_dir) → DataFrame`

Join labels with available features (inner join on `slide_id`).

**Parameters:**
- `labels_df`: DataFrame with `slide_id`, `label` columns
- `features_dir`: Directory containing `.h5` files

**Returns:**
- DataFrame containing only slides with both labels AND features

**Side Effects:**
- Prints summary of matched/missing slides to stdout

---

### 4.3 Layer 3: Interface

#### `class MILDataset`

Unified dataset combining labels and features.

**Constructor:**

```python
MILDataset(labels_csv: str, features_dir: str)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `slide_ids` | `List[str]` | All slide IDs in dataset |
| `labels` | `List[int]` | All labels in dataset |
| `embed_dim` | `int` | Embedding dimension |
| `num_classes` | `int` | Number of unique classes |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `__len__()` | `int` | Number of slides |
| `__iter__()` | `Iterator[SlideData]` | Iterate over slides |
| `__getitem__(key)` | `SlideData` | Get by index (int) or slide_id (str) |
| `get_subset(slide_ids)` | `MILDataset` | Create subset with specific slides |
| `split_by_column(column)` | `Dict[str, MILDataset]` | Split by column values |

---

#### `class SlideData`

Container for a single slide's data.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `slide_id` | `str` | Slide identifier |
| `features` | `Tensor` | Shape `[M, D]` |
| `label` | `Optional[int]` | Class label |
| `case_id` | `Optional[str]` | Patient/case identifier |

---

### 4.4 Tensor Preparation

#### `prepare_for_mil(features) → Tensor`

Add batch dimension for MIL model input.

**Parameters:**
- `features`: Tensor of shape `[M, D]`

**Returns:**
- Tensor of shape `[1, M, D]`

---

#### `batch_for_mil(features_list) → Tuple[Tensor, Tensor]`

Batch multiple slides with padding.

**Parameters:**
- `features_list`: List of tensors, each `[M_i, D]`

**Returns:**
- `padded`: Tensor of shape `[B, M_max, D]`
- `mask`: Tensor of shape `[B, M_max]` (1=valid, 0=padding)

---

## 5. Data Flow

### 5.1 Initialization

```
labels.csv ──► load_labels() ──► labels_df
                                     │
features_dir ──► get_available_features() ──► features_df
                                                   │
                         join_labels_and_features()
                                     │
                                     ▼
                               matched_df
                                     │
                                     ▼
                               MILDataset
```

### 5.2 Data Access

```
MILDataset[slide_id]
        │
        ▼
  load_features(h5_path)
        │
        ▼
   SlideData(slide_id, features, label, case_id)
        │
        ▼
  prepare_for_mil(features)
        │
        ▼
   [1, M, D] tensor ──► MIL Model
```

---

## 6. Error Handling

### 6.1 Missing Data

| Scenario | Behavior |
|----------|----------|
| Slide in CSV but no H5 file | Excluded from dataset, logged |
| H5 file but no CSV entry | Excluded from dataset, logged |
| Empty features directory | `FileNotFoundError` |
| Missing required CSV columns | `ValueError` |

### 6.2 Shape Validation

| Input Shape | Output Shape | Action |
|-------------|--------------|--------|
| `(M, D)` | `(M, D)` | None |
| `(1, M, D)` | `(M, D)` | Squeeze dim 0 |
| `(D,)` | `(1, D)` | Unsqueeze dim 0 |

---

## 7. Constraints

### 7.1 Invariants

1. All H5 files in a dataset MUST have the same embedding dimension
2. `slide_id` MUST be unique within a dataset
3. `label` MUST be non-negative integers
4. Features tensor MUST be 2D after normalization

### 7.2 Assumptions

1. H5 files are in CLAM output format
2. Labels CSV uses comma as delimiter
3. Feature files are not modified during dataset lifetime
4. Memory is sufficient to hold one slide's features at a time

---

## 8. Future Extensions

### 8.1 Planned

| Feature | Priority | Description |
|---------|----------|-------------|
| PyTorch DataLoader integration | High | Batching, shuffling, workers |
| Weighted sampling | High | Class imbalance handling |
| Feature caching | Medium | LRU cache for repeated access |
| Lazy loading | Medium | Don't read H5 until accessed |
| Embedding dimension validation | Medium | Verify all H5s match |

### 8.2 Dagster Integration

Each function maps to an asset:

```python
@asset
def labels() -> pd.DataFrame:
    return load_labels(config.labels_csv)

@asset
def feature_manifest() -> pd.DataFrame:
    return get_available_features(config.features_dir)

@asset
def matched_slides(labels, feature_manifest) -> pd.DataFrame:
    return join_labels_and_features(labels, config.features_dir)

@asset
def dataset(matched_slides) -> MILDataset:
    return MILDataset(config.labels_csv, config.features_dir)
```

---

## 9. References

- CLAM: https://github.com/mahmoodlab/CLAM
- MIL-Lab: https://github.com/mahmoodlab/MIL-Lab
- HDF5 Specification: https://www.hdfgroup.org/solutions/hdf5/

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-15 | — | Initial specification |
