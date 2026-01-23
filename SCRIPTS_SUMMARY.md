# MIL-Lab Scripts Summary

This document provides a comprehensive overview of all scripts in the top-level directory of the MIL-Lab repository.

---

## Table of Contents

1. [Experiment Runner Scripts](#experiment-runner-scripts)
2. [Training Scripts](#training-scripts)
3. [Heatmap Generation Scripts](#heatmap-generation-scripts)
4. [Visualization Scripts](#visualization-scripts)
5. [Visualization Directory (NEW)](#visualization-directory)
6. [Debugging and Testing Scripts](#debugging-and-testing-scripts)
7. [Utility Scripts](#utility-scripts)
8. [Shell Scripts](#shell-scripts)
9. [Documentation Files](#documentation-files)

---

## Experiment Runner Scripts

These scripts automate running multiple MIL model experiments with different configurations.

### `run_mil_experiments.py`
**Purpose**: Run multiple MIL model and encoder combinations with single train/val/test split

**Key Features**:
- Single 80/10/10 train/val/test split
- Supports all MIL models (ABMIL, TransMIL, CLAM, DSMIL, DFTD, etc.)
- Automatic feature dimension handling per encoder
- Generates metrics CSV and confusion matrices
- GPU execution with mixed precision training (FP16)
- Early stopping with validation monitoring
- Weighted sampling for class imbalance

**Output**:
- Experiment metrics CSV
- Confusion matrices for each experiment
- Trained model weights
- Configuration log

**Documentation**: See `EXPERIMENT_RUNNER_README.md`

### `run_mil_experiments_cv.py`
**Purpose**: Run multiple MIL model and encoder combinations with K-fold cross-validation

**Key Features**:
- K-fold cross-validation (default: 5-fold)
- Reports mean ± std metrics across folds
- Stratified splitting maintains class distribution
- Two CSV outputs: summary stats and per-fold details
- Per-fold confusion matrices and model weights
- All features from `run_mil_experiments.py`

**Output**:
- Summary metrics CSV with mean ± std
- Per-fold detailed metrics CSV
- Confusion matrix for each fold
- Model weights for each fold
- Experiment configuration

**Documentation**: See `CV_EXPERIMENT_README.md`

### `run_mil_experiments_predefined_splits.py`
**Purpose**: Run MIL experiments using predefined train/test splits from PathBench

**Key Features**:
- Uses `k=all.tsv` for predefined train/test splits
- Optionally creates validation set from training data
- No random splitting - reproducible results
- Compatible with PathBench benchmark datasets
- All experiment features from base runner

**Configuration**:
- `TSV_PATH`: Path to predefined splits file
- `FOLD_COLUMN`: Column name containing split info (e.g., 'fold_0')
- `VAL_FRACTION`: Fraction of training data for validation (0-1)

**Documentation**: See `PREDEFINED_SPLITS_README.md`

### `run_mil_experiments_gem3.py`
**Purpose**: Specialized experiment runner aligned with "Do Multiple Instance Learning Models Transfer?" paper

**Key Features**:
- Corrected to match paper 2506.09022v2 replication details
- Uses specific UNI patient splits (not random)
- WeightedRandomSampler for class imbalance handling
- Paper-aligned training configuration

**Use Case**: Reproducing results from transfer learning paper

### `run_mil.py`
**Purpose**: Simplified single MIL training run with shared utilities

**Key Features**:
- Train a single model configuration
- Uses shared utilities from `utils/`
- Includes UNI feature normalization
- Early stopping and dropout
- Final corrected version matching paper specifications

**Configuration**: Edit constants at top of file for paths and hyperparameters

---

## Training Scripts

Simple training scripts for specific MIL models.

### `train_abmil_simple.py`
**Purpose**: Simplified ABMIL training using shared utilities

**Key Features**:
- Direct ABMIL model training
- Configurable via constants at top of file
- Basic training loop with evaluation
- Confusion matrix generation

**Configuration**:
- `GRADE_GROUP`: Use grade groups (True) or ISUP scores (False)
- `EXCLUDE_MID_GRADE`: Exclude mid-grade samples
- Early stopping and dropout options

### `train_abmil_feather_panda.py`
**Purpose**: ABMIL training for PANDA dataset with Feather River data source

**Key Features**:
- Specialized for Feather River prostate cancer data
- PANDA-specific preprocessing
- Full training pipeline

### `train_clam_simple.py`
**Purpose**: Simplified CLAM training using shared utilities

**Key Features**:
- CLAM model training with instance-level learning
- Uses shared utilities from `utils/`
- Configurable hyperparameters

### `train_clam_panda.py`
**Purpose**: Full CLAM training pipeline for PANDA dataset

**Key Features**:
- Complete CLAM implementation
- PANDA dataset specific
- Instance-level and bag-level training

### `fixed_abmil.py`
**Purpose**: Fixed ABMIL implementation addressing known issues

**Key Features**:
- Bug fixes for ABMIL model
- Corrected attention mechanism
- May contain experimental fixes

---

## Heatmap Generation Scripts

Scripts for generating attention heatmaps from trained MIL models.

### `generate_heatmaps.py`
**Purpose**: Generate attention heatmaps from trained MIL models

**Key Features**:
- Optimized for new H5 embedding format: `(num_patches, feature_dim)`
- Supports any MIL model with attention mechanism
- Generates heatmaps overlaid on whole slide images
- Command-line interface

**Usage**:
```bash
python generate_heatmaps.py \
    --checkpoint path/to/model.pt \
    --h5_dir path/to/embeddings/ \
    --slide_dir path/to/slides/ \
    --output_dir heatmaps/output/
```

**Output**: Heatmap images showing attention scores per patch

**Documentation**: See `HEATMAPS_README.md`, `HEATMAP_QUICKSTART.md`

### `generate_heatmaps_highres.py`
**Purpose**: Generate high-resolution heatmaps with overlapping patches

**Key Features**:
- Dense grid of overlapping patches (90-95% overlap)
- On-the-fly feature extraction
- High-resolution output at 20x or any resolution
- Smooth, detailed attention visualizations

**Usage**:
```bash
python generate_heatmaps_highres.py \
    --checkpoint path/to/model.pt \
    --slide path/to/slide.tiff \
    --output heatmap_highres.tiff \
    --overlap 0.90 \
    --vis_level 0
```

**Documentation**: See `HIGHRES_HEATMAPS.md`

### `create_heatmaps.py`
**Purpose**: Full-featured heatmap generation with YAML configuration

**Key Features**:
- YAML configuration file support
- Compatible with OpenSlide for synchronized viewing
- Inspired by CLAM framework
- Adapted for MIL-Lab

**Usage**:
```bash
python create_heatmaps.py --config heatmaps/configs/config.yaml
```

**Documentation**: See `HEATMAP_FILES_SUMMARY.md`, `HEATMAP_GENERATION.md`

---

## Visualization Scripts

Scripts for visualizing model predictions and attention.

### `visualize_model.py`
**Purpose**: Standalone visualization for MIL model predictions

**Key Features**:
- Works with CLAM and ABMIL models
- Visualize top patches by attention
- Generate heatmaps
- Command-line interface

**Usage**:
```bash
python visualize_model.py \
    --model-type abmil \
    --model-path model.pt \
    --csv-path train.csv \
    --feats-path features/ \
    --wsi-dir slides/ \
    --num-classes 3
```

### `view_heatmap_sync.py`
**Purpose**: View heatmaps synchronized with whole slide images

**Key Features**:
- OpenSlide integration
- Side-by-side viewing
- Interactive navigation

**Documentation**: See `HEATMAP_SUMMARY.md`

### `create_process_list.py`
**Purpose**: Create processing lists for batch heatmap generation

**Key Features**:
- Generate CSV lists of slides to process
- Filter by dataset criteria
- Organize batch processing jobs

---

## Visualization Directory

**NEW**: A dedicated directory for organized heatmap generation with JSON configuration.

### Location: `Visualization/`

This directory contains the refactored high-resolution heatmap generation pipeline with JSON-based configuration.

### Contents

- **`generate_heatmaps_highres.py`**: JSON-configurable high-resolution heatmap generator
- **`config_example.json`**: Standard quality configuration (90% overlap)
- **`config_high_overlap.json`**: Ultra high quality (95% overlap)
- **`config_fast.json`**: Fast preview configuration (70% overlap)
- **`README.md`**: Comprehensive documentation

### Key Features

- **JSON Configuration**: All parameters in easy-to-edit JSON files
- **Multiple Presets**: Standard, high-quality, and fast preview configs
- **Auto-detection**: Automatic feature extractor detection from checkpoint
- **Batch Processing**: Easy to script for multiple slides
- **Organized Output**: Separate directory keeps visualization code clean

### Usage

```bash
cd Visualization
python generate_heatmaps_highres.py --config config_example.json
```

### Example Configuration

```json
{
  "checkpoint": "/path/to/model.pt",
  "slide": "/path/to/slide.tiff",
  "output": "/path/to/heatmap.tiff",

  "model_name": "abmil.base.uni_v2.pc108-24k",
  "num_classes": 2,
  "feature_extractor": "auto",

  "overlap": 0.90,
  "patch_size": 256,
  "vis_level": 0,

  "cmap": "jet",
  "alpha": 0.5
}
```

### Configuration Presets

| Preset | Overlap | Use Case | Processing Time |
|--------|---------|----------|-----------------|
| Standard (`config_example.json`) | 90% | Production quality | ~5-10 min |
| High Quality (`config_high_overlap.json`) | 95% | Publication quality | ~15-30 min |
| Fast (`config_fast.json`) | 70% | Quick preview | ~1-2 min |

### Advantages Over Command-Line Version

1. **Reproducibility**: Save and version control exact configurations
2. **Batch Processing**: Easy to process multiple slides programmatically
3. **Organization**: Clean separation from other scripts
4. **Documentation**: Comprehensive README in the directory
5. **No Long Commands**: No more 10+ argument command lines

**Documentation**: See `Visualization/README.md` for full details

---

## Debugging and Testing Scripts

Scripts for debugging and testing functionality.

### `debug_h5_structure.py`
**Purpose**: Diagnostic script to check H5 file structure

**Key Features**:
- Inspect H5 file contents
- Check tensor shapes
- Verify data integrity
- Reports on first 5 files in directory

**Usage**: Edit paths in script and run directly

### `debug_dataset_loading.py`
**Purpose**: Debug dataset loading issues

**Key Features**:
- Test dataset loading pipeline
- Verify data preprocessing
- Check batch creation

### `debug_heatmap.py`
**Purpose**: Debug heatmap generation issues

**Key Features**:
- Test heatmap pipeline
- Verify attention extraction
- Check visualization rendering

### `test_heatmap_setup.py`
**Purpose**: Test heatmap generation setup

**Key Features**:
- Verify dependencies
- Test file paths
- Check model compatibility

### `test_simple_heatmap.py`
**Purpose**: Simple heatmap generation test

**Key Features**:
- Minimal test case
- Quick verification
- Debugging aid

### `test_model_shapes.py`
**Purpose**: Test MIL model input/output shapes

**Key Features**:
- Debug shape mismatches
- Test different batch sizes
- Verify model compatibility

**Output**: Prints shape information for debugging

### `test_predefined_splits.py`
**Purpose**: Test predefined split loading functionality

**Key Features**:
- Verify TSV split file parsing
- Check split distribution
- Validate data loading

---

## Utility Scripts

General utility scripts for data management and validation.

### `validate_embeddings.py`
**Purpose**: Validate H5 embedding files and compare directories

**Key Features**:
- Check H5 file integrity
- Compare old vs new feature directories
- Report file statistics (shapes, dtypes)
- Identify missing or corrupted files

**Configuration**: Edit paths at top of file

**Output**: Validation report with errors and warnings

### `check_patch_counts.py`
**Purpose**: Quality control for H5 feature files

**Key Features**:
- Check patch counts in all H5 files
- Report slides with insufficient patches (<24 by default)
- Generate CSV report with slide IDs and patch counts
- Useful for identifying slides needing reprocessing

**Usage**:
```bash
python check_patch_counts.py \
    --feats-path path/to/features/ \
    --min-patches 24 \
    --csv-path train.csv \
    --output-csv patch_report.csv
```

### `compare_labels.py`
**Purpose**: Compare labels between different datasets or splits

**Key Features**:
- Verify label consistency
- Check class distribution
- Identify labeling discrepancies

---

## Shell Scripts

Bash scripts for common operations.

### `example_generate_heatmaps.sh`
**Purpose**: Example script for batch heatmap generation

**Key Features**:
- Shows typical usage pattern
- Loop over multiple slides
- Configurable parameters

**Usage**: Edit paths and run `./example_generate_heatmaps.sh`

### `example_highres_heatmap.sh`
**Purpose**: Example script for high-resolution heatmap generation

**Key Features**:
- Demonstrates high-res heatmap workflow
- Overlapping patch configuration
- Sample command structure

---

## Documentation Files

Comprehensive documentation for various components.

### Experiment Documentation
- **`EXPERIMENT_RUNNER_README.md`**: Guide for single-split experiment runner
- **`CV_EXPERIMENT_README.md`**: Guide for cross-validation experiments
- **`PREDEFINED_SPLITS_README.md`**: Guide for using predefined splits
- **`TRAINING_SCRIPTS_README.md`**: Overview of training scripts

### Heatmap Documentation
- **`HEATMAPS_README.md`**: Main heatmap generation guide
- **`HEATMAP_QUICKSTART.md`**: Quick start guide for heatmaps
- **`HEATMAP_GENERATION.md`**: Detailed heatmap generation process
- **`HEATMAP_FILES_SUMMARY.md`**: Summary of heatmap-related files
- **`HEATMAP_SUMMARY.md`**: High-level heatmap overview
- **`HEATMAP_CHECKLIST.md`**: Checklist for heatmap generation
- **`HIGHRES_HEATMAPS.md`**: Guide for high-resolution heatmaps

### General Documentation
- **`README.md`**: Main repository README with project overview

---

## Quick Reference Guide

### To Run Experiments

**Single experiment with train/val/test split**:
```bash
python run_mil_experiments.py
# Configure experiments in EXPERIMENTS list at top of file
```

**Cross-validation experiments**:
```bash
python run_mil_experiments_cv.py
# Configure K-fold and experiments in file
```

**Using predefined splits**:
```bash
python run_mil_experiments_predefined_splits.py
# Set TSV_PATH and FOLD_COLUMN in file
```

### To Generate Heatmaps

**Recommended: Use Visualization directory with JSON config** (NEW):
```bash
cd Visualization
# Edit config file with your paths
python generate_heatmaps_highres.py --config config_example.json
```

**Standard heatmaps**:
```bash
python generate_heatmaps.py \
    --checkpoint models/model.pt \
    --h5_dir features/ \
    --slide_dir slides/ \
    --output_dir heatmaps/
```

**High-resolution heatmaps (legacy command-line)**:
```bash
python generate_heatmaps_highres.py \
    --checkpoint models/model.pt \
    --slide slide.tiff \
    --output heatmap.tiff \
    --overlap 0.90
```

### To Train Models

**Simple ABMIL training**:
```bash
python train_abmil_simple.py
# Edit configuration constants at top of file
```

**Simple CLAM training**:
```bash
python train_clam_simple.py
# Edit configuration constants at top of file
```

### To Debug

**Check H5 file structure**:
```bash
python debug_h5_structure.py
```

**Validate embeddings**:
```bash
python validate_embeddings.py
```

**Check patch counts**:
```bash
python check_patch_counts.py --feats-path features/ --min-patches 24
```

**Test model shapes**:
```bash
python test_model_shapes.py
```

---

## File Organization Best Practices

1. **Experiment Results**: All experiment runners create timestamped directories in `experiment_results/` or `experiment_results_predefined/`

2. **Heatmaps**: Generated heatmaps go to `heatmaps/output/` by default

3. **Model Checkpoints**: Saved in experiment output directories with descriptive names

4. **Documentation**: All markdown files are in the root directory for easy access

5. **Configuration**: Most scripts use constants at the top of the file or command-line arguments

---

## Common Workflow Examples

### Complete Experiment Workflow
1. Prepare data and features
2. Run experiment runner: `python run_mil_experiments.py`
3. Review metrics CSV and confusion matrices
4. Select best model checkpoint
5. Generate heatmaps: `python generate_heatmaps.py --checkpoint best_model.pt ...`
6. Visualize results: `python visualize_model.py --model-path best_model.pt ...`

### Debugging Workflow
1. Check H5 files: `python debug_h5_structure.py`
2. Validate embeddings: `python validate_embeddings.py`
3. Check patch counts: `python check_patch_counts.py`
4. Test model: `python test_model_shapes.py`
5. Debug heatmap: `python debug_heatmap.py`

### Quality Control Workflow
1. Validate embeddings across dataset
2. Check patch counts to find problematic slides
3. Remove or reprocess slides with insufficient patches
4. Compare labels between splits
5. Run test experiments to verify setup

---

## Key Dependencies

Most scripts depend on:
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **H5py**: Feature file handling
- **scikit-learn**: Metrics and splitting
- **OpenSlide**: Whole slide image handling
- **PIL**: Image processing
- **tqdm**: Progress bars

Custom modules:
- **`src.builder`**: Model creation
- **`src.visualization`**: Heatmap utilities
- **`utils`**: Shared data processing utilities

---

## Notes

- Most scripts require editing configuration constants at the top of the file
- GPU is strongly recommended for training and inference
- H5 feature files should be in format `(num_patches, feature_dim)`
- Model checkpoints contain full model state including weights and configuration
- Experiment runners create timestamped output directories automatically

---

**Last Updated**: 2025-12-05
