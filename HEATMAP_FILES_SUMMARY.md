# Heatmap Visualization - Files Summary

This document provides an overview of all files created for the heatmap generation system.

## Core Scripts

### `create_heatmaps.py`
**Purpose:** Main script for generating attention heatmaps from trained MIL models

**Key Features:**
- Loads trained model from checkpoint
- Runs inference to get attention scores
- Generates heatmap visualizations overlaid on WSI
- Saves results in OpenSlide-compatible format
- Optionally samples top-k patches

**Usage:**
```bash
python create_heatmaps.py --config heatmaps/configs/my_config.yaml
```

### `view_heatmap_sync.py`
**Purpose:** Interactive viewer for synchronized viewing of WSI and heatmap

**Key Features:**
- Side-by-side display of original slide and heatmap
- Synchronized navigation (same region in both views)
- Interactive sliders for position and zoom level
- Support for multi-resolution pyramid levels

**Usage:**
```bash
python view_heatmap_sync.py --slide path/to/slide.svs --heatmap path/to/heatmap.png
```

### `create_process_list.py`
**Purpose:** Helper script to generate CSV listing slides to process

**Key Features:**
- Scans directory for slide files
- Optionally matches with feature files
- Adds labels from external CSV
- Marks slides with missing features

**Usage:**
```bash
python create_process_list.py \
    --slide_dir /path/to/slides \
    --features_dir /path/to/features \
    --output heatmaps/process_lists/my_list.csv
```

## Visualization Utilities

### `src/visualization/heatmap_utils.py`
**Purpose:** Core heatmap generation utilities

**Key Components:**
- `WholeSlideImage`: WSI wrapper with tissue segmentation
- `draw_heatmap()`: Create heatmap overlay on WSI
- `sample_top_patches()`: Extract high-attention patches
- `MILLabVisualizer`: High-level visualization interface
- `to_percentiles()`: Score normalization

**Features:**
- CLAM-inspired implementation
- Tissue segmentation with hole detection
- Percentile normalization for better contrast
- Multiple colormap support
- Gaussian blur smoothing
- Alpha blending

### `src/visualization/__init__.py`
**Purpose:** Package initialization and exports

**Exports:**
- `MILLabVisualizer`
- `WholeSlideImage`
- `draw_heatmap`
- `sample_top_patches`
- `to_percentiles`

## Utility Modules

### `utils/file_utils.py`
**Purpose:** File I/O utilities for HDF5 and pickle files

**Functions:**
- `save_hdf5()`: Save attention scores and coordinates to HDF5
- `load_hdf5()`: Load data from HDF5 files
- `save_pkl()`: Save objects with pickle
- `load_pkl()`: Load pickled objects
- `get_slide_id()`: Extract slide ID from file path

## Configuration Files

### `heatmaps/configs/config_template.yaml`
**Purpose:** Generic configuration template for any dataset/model

**Sections:**
- `exp_arguments`: Experiment settings (name, output directories)
- `data_arguments`: Data paths (slides, features, labels)
- `patching_arguments`: Patch parameters (size, overlap, level)
- `model_arguments`: Model architecture and checkpoint
- `heatmap_arguments`: Visualization parameters
- `sample_arguments`: Patch sampling configuration

### `heatmaps/configs/config_abmil_panda.yaml`
**Purpose:** Example configuration for PANDA dataset with AB-MIL

**Customizations:**
- PANDA-specific paths (.tiff extension)
- Binary classification (normal vs tumor)
- UNI features (in_dim: 1024)
- Recommended visualization settings

## Documentation

### `HEATMAP_GENERATION.md` (15 KB, comprehensive)
**Purpose:** Complete documentation for heatmap generation

**Contents:**
1. Overview and pipeline description
2. Quick start guide
3. Detailed configuration reference
4. Advanced usage (ROI, multi-directory, etc.)
5. Viewing with OpenSlide
6. Helper scripts and workflows
7. Troubleshooting guide
8. Output format specifications
9. Integration with existing workflows
10. Citation information

### `HEATMAP_QUICKSTART.md` (8 KB, concise)
**Purpose:** 5-minute quick start guide

**Contents:**
1. Prerequisites checklist
2. 3-step process (list → config → generate)
3. Viewing options
4. Customization examples
5. Common issues and solutions
6. Output descriptions
7. Next steps and workflows

### `heatmaps/README.md` (5 KB, reference)
**Purpose:** Quick reference guide for heatmap generation

**Contents:**
1. Quick start commands
2. Directory structure
3. Essential config parameters
4. Common use cases
5. Troubleshooting table
6. Output file descriptions
7. Examples

### `HEATMAP_FILES_SUMMARY.md` (this file)
**Purpose:** Overview of all files in the heatmap system

## Example Files

### `heatmaps/process_lists/example_process_list.csv`
**Purpose:** Example process list showing CSV format

**Format:**
```csv
slide_id,label,process,features_path,coords_path
slide_001,tumor,1,features/h5_files/slide_001.pt,features/h5_files/slide_001.h5
```

**Columns:**
- `slide_id`: Slide identifier
- `label`: Class label (optional)
- `process`: 1=process, 0=skip
- `features_path`: Path to feature file (optional)
- `coords_path`: Path to coordinates (optional)

## Output Directory Structure

### During/After Heatmap Generation

```
heatmaps/
├── configs/                           # Configuration files (input)
│   ├── config_template.yaml
│   └── config_abmil_panda.yaml
│
├── process_lists/                     # Slide lists (input)
│   └── example_process_list.csv
│
├── heatmap_raw_results/              # Raw outputs
│   └── {experiment_name}/
│       └── {label}/
│           └── {slide_id}/
│               ├── {slide_id}_blockmap.h5        # Attention + coords
│               └── ...
│
├── heatmap_production_results/       # Final visualizations
│   └── {experiment_name}/
│       └── {label}/
│           ├── {slide_id}_heatmap.png           # Heatmap overlay
│           ├── {slide_id}_original.png          # Original H&E
│           └── sampled_patches/                  # Top-k patches
│               └── label_{label}_pred_{pred}/
│                   └── {sample_name}/
│                       ├── 0_{slide}_x_{x}_y_{y}_score_{s}.png
│                       └── ...
│
└── results/                           # Results summaries
    └── {experiment_name}_results.csv
```

## File Dependencies

### Runtime Dependencies
```
create_heatmaps.py
├── requires: src/visualization/heatmap_utils.py
├── requires: src/builder.py
├── requires: utils/file_utils.py
└── requires: PyYAML, torch, h5py, pandas, PIL, tqdm

view_heatmap_sync.py
├── requires: openslide-python
├── requires: matplotlib
└── requires: PIL, numpy

create_process_list.py
├── requires: pandas
└── requires: pathlib
```

### Installation Requirements
```bash
# Core dependencies (should already be installed)
pip install torch torchvision
pip install h5py pandas tqdm pyyaml

# Visualization dependencies (new)
pip install openslide-python
pip install matplotlib pillow
pip install scipy  # for percentile calculations
```

## Integration with MIL-Lab

### Existing Components Used
- `src/builder.py`: Model initialization
- `src/models/`: Model architectures (ABMIL, CLAM, etc.)
- Feature extraction outputs: `.pt` and `.h5` files

### New Components Added
- `src/visualization/`: Visualization utilities
- `utils/file_utils.py`: File I/O utilities
- `create_heatmaps.py`: Main generation script
- `view_heatmap_sync.py`: Interactive viewer
- `create_process_list.py`: List generation helper

## Typical Workflow

```
1. Feature Extraction (existing)
   └── extract_features_fp.py
       ├── Output: features/h5_files/{slide_id}.pt
       └── Output: features/h5_files/{slide_id}.h5

2. Model Training (existing)
   └── train_abmil.py / train_clam.py
       └── Output: checkpoints/{experiment}.pt

3. Process List Creation (new)
   └── create_process_list.py
       └── Output: heatmaps/process_lists/{list}.csv

4. Heatmap Generation (new)
   └── create_heatmaps.py
       ├── Input: trained model checkpoint
       ├── Input: features and coordinates
       ├── Input: configuration YAML
       ├── Output: heatmap visualizations
       └── Output: attention scores + results CSV

5. Visualization (new)
   └── view_heatmap_sync.py
       ├── Input: original WSI
       ├── Input: generated heatmap
       └── Interactive viewer
```

## File Sizes (Approximate)

### Source Code
- `create_heatmaps.py`: ~15 KB
- `view_heatmap_sync.py`: ~12 KB
- `create_process_list.py`: ~8 KB
- `src/visualization/heatmap_utils.py`: ~19 KB
- `utils/file_utils.py`: ~4 KB

### Documentation
- `HEATMAP_GENERATION.md`: ~15 KB
- `HEATMAP_QUICKSTART.md`: ~8 KB
- `heatmaps/README.md`: ~5 KB

### Configurations
- `config_template.yaml`: ~2 KB
- `config_abmil_panda.yaml`: ~2 KB

### Total: ~90 KB of code and documentation

## Summary

The heatmap visualization system consists of:

✅ **3 main scripts** (create, view, process-list)
✅ **2 utility modules** (visualization, file I/O)
✅ **2 config templates** (generic + PANDA)
✅ **4 documentation files** (comprehensive + quick + reference + summary)
✅ **1 example file** (process list CSV)

All files are CLAM-compatible and integrate seamlessly with the existing MIL-Lab framework.

## Quick Reference

| Task | File | Command |
|------|------|---------|
| Generate heatmaps | `create_heatmaps.py` | `python create_heatmaps.py --config CONFIG` |
| View heatmaps | `view_heatmap_sync.py` | `python view_heatmap_sync.py --slide SLIDE --heatmap HEATMAP` |
| Create slide list | `create_process_list.py` | `python create_process_list.py --slide_dir DIR --output CSV` |
| Learn usage | `HEATMAP_QUICKSTART.md` | Open in editor |
| Full reference | `HEATMAP_GENERATION.md` | Open in editor |
| Quick tips | `heatmaps/README.md` | Open in editor |

---

**Last Updated:** 2024-11-17
**Version:** 1.0
**Compatible with:** MIL-Lab, CLAM, OpenSlide
