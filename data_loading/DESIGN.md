# Data Loading Module - Design Specification

## Overview

The `data_loading` module provides a flexible data loading infrastructure for Multiple Instance Learning (MIL) that supports different levels of data aggregation.

## Architecture

```
                    ┌─────────────────┐
                    │   Labels CSV    │
                    │ slide_id, label │
                    │ case_id, split  │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────┐     ┌─────────────────┐
│  H5 Files   │────▶│   MILDataset    │  (slide-level)
│ features/   │     │  One bag per H5 │
└─────────────┘     └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Use as-is  │  │ .concat_by()│  │ .group_by() │
    │ (PANDA)     │  │             │  │             │
    └─────────────┘  └──────┬──────┘  └──────┬──────┘
                            │                │
                            ▼                ▼
                   ┌────────────────┐ ┌──────────────────┐
                   │GroupedMILDataset│ │HierarchicalMILDataset│
                   │ Flat concat    │ │ Preserves structure │
                   └────────────────┘ └──────────────────┘
```

## Components

### Data Containers

| Class | Description | Features Shape |
|-------|-------------|----------------|
| `SlideData` | Single slide | `[M, D]` |
| `GroupedData` | Grouped items (flat) | `[Σ M_i, D]` |
| `HierarchicalData` | Grouped items (structured) | `List[Tensor]` |

### Dataset Classes

#### `MILDataset` (Base)
- **Purpose**: Slide-level MIL dataset
- **Use case**: One bag per H5 file (e.g., PANDA where 1 patient = 1 slide)
- **Methods**:
  - `.concat_by(column)` → `GroupedMILDataset`
  - `.group_by(column)` → `HierarchicalMILDataset`
  - `.split_by_column(column)` → Dict of subsets
  - `.random_split(...)` → Dict of subsets

#### `GroupedMILDataset` (Flat Concatenation)
- **Purpose**: Group items and concatenate features
- **Use case**: Multi-slide patients where slide boundaries don't matter
- **Created via**: `MILDataset.concat_by('case_id')`
- **Features**: All patches from all slides concatenated into one bag

#### `HierarchicalMILDataset` (Preserves Structure)
- **Purpose**: Group items but preserve internal structure
- **Use case**: Two-stage attention (patch→slide→patient)
- **Created via**: `MILDataset.group_by('case_id')`
- **Features**: List of tensors, one per slide

## Design Decisions

### 1. Method-Based Grouping vs Separate Classes

**Chosen**: Method-based (`.concat_by()`, `.group_by()`)

**Rationale**:
- Single entry point (`MILDataset`)
- No code duplication
- Clear transformation path
- Easy to understand: "start with slides, then group if needed"

### 2. Flat vs Hierarchical

**Problem**: Should multi-slide patients be one big bag or preserve structure?

**Solution**: Support both:
- `.concat_by()` for flat (simpler, works for most cases)
- `.group_by()` for hierarchical (enables two-stage attention)

### 3. Data Loading vs Model Aggregation

**Principle**: Data loading provides data with metadata. Model decides how to aggregate.

- `GroupedMILDataset`: Data layer does concatenation
- `HierarchicalMILDataset`: Data layer preserves structure, model handles aggregation

## Usage Examples

### Slide-Level (PANDA)
```python
dataset = MILDataset('labels.csv', 'features/')
splits = dataset.split_by_column('split')
train_loader, adapter = create_dataloader(splits['train'])
```

### Patient-Level (Flat)
```python
dataset = MILDataset('labels.csv', 'features/')
patient_dataset = dataset.concat_by('case_id')
splits = patient_dataset.random_split(stratify=True)
```

### Patient-Level (Hierarchical)
```python
dataset = MILDataset('labels.csv', 'features/')
patient_dataset = dataset.group_by('case_id')

# Each sample returns HierarchicalData with features as List[Tensor]
patient = patient_dataset[0]
for slide_features in patient.features:
    print(slide_features.shape)  # [M_i, D]
```

## CSV Format

Required columns:
- `slide_id`: Unique identifier matching H5 filename
- `label`: Class label

Optional columns:
- `case_id`: Patient/case identifier (required for grouping)
- `split`: Predefined train/val/test split

## Future Extensions

1. **Core-level support**: If H5 files are per-core, group by `slide_id` then `case_id`
2. **Lazy loading**: Load features on-demand for large datasets
3. **Caching**: Cache loaded features in memory
4. **Multi-label**: Support for multi-label classification
