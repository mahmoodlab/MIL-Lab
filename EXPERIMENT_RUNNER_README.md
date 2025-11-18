# MIL Experiment Runner

This guide explains how to use `run_mil_experiments.py` to run multiple MIL model and encoder combinations.

## Overview

The experiment runner allows you to:
- Run multiple MIL model + encoder combinations in sequence
- Automatically generate metrics CSV for all experiments
- Create confusion matrices for each experiment
- Track all results in timestamped output directories

## Quick Start

### 1. Configure Your Experiments

Edit `run_mil_experiments.py` and modify the `EXPERIMENTS` list:

```python
EXPERIMENTS = [
    # Format: (model_config, display_name)
    ('abmil.base.gigapath.pc108-24k', 'ABMIL + GigaPath'),
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('abmil.base.conch_v15.pc108-24k', 'ABMIL + CONCH_v1.5'),
    # Add more combinations:
    ('transmil.base.gigapath.pc108-24k', 'TransMIL + GigaPath'),
    ('clam.base.uni_v2.pc108-24k', 'CLAM + UNI_v2'),
]
```

### 2. Configure Feature Paths

Update the `FEATURE_PATHS` dictionary with your encoder feature paths:

```python
FEATURE_PATHS = {
    'gigapath': '/path/to/features_gigapath/',
    'uni_v2': '/path/to/features_uni_v2/',
    'conch_v15': '/path/to/features_conch_v15/',
    # Add more as needed
}
```

### 3. Run Experiments

```bash
python run_mil_experiments.py
```

## Available Models

Based on `src/_global_mappings.py`, the following MIL models are available:

- `abmil` - Attention-Based MIL
- `transmil` - Transformer MIL
- `transformer` - Standard Transformer
- `dftd` - DFTD
- `clam` - CLAM
- `ilra` - ILRA
- `rrt` - RRT-MIL
- `wikg` - WIKG-MIL
- `dsmil` - DS-MIL

## Available Encoders

Supported encoders (from `src/_global_mappings.py`):

- `uni` (1024 dim)
- `uni_v2` (1536 dim)
- `ctranspath` (768 dim)
- `conch` (512 dim)
- `conch_v15` (768 dim)
- `gigapath` (1536 dim)
- `resnet50` (1024 dim)
- `virchow` (2560 dim)
- `virchow2` (2560 dim)
- `phikon` (768 dim)
- `phikon2` (1024 dim)
- `hoptimus` (1536 dim)
- `hoptimus1` (1536 dim)
- `musk` (1024 dim)

## Model Config String Format

Model configs follow this pattern:
```
{model_name}.{config}.{encoder}.{pretrained_dataset}
```

Examples:
- `abmil.base.uni_v2.pc108-24k` - ABMIL with base config, UNI_v2 encoder, PC-108 pretrained
- `transmil.base.gigapath.pc108-24k` - TransMIL with base config, GigaPath encoder, PC-108 pretrained
- `clam.base.conch_v15.pc108-24k` - CLAM with base config, CONCH_v1.5 encoder, PC-108 pretrained

## Output Structure

```
experiment_results/
└── run_20250117_143022/
    ├── experiment_config.json          # Configuration used
    ├── experiment_results.csv          # All metrics in CSV
    ├── best_model_abmil_gigapath.pth  # Trained model weights
    ├── confusion_matrix_abmil_gigapath.png
    ├── best_model_abmil_uni_v2.pth
    ├── confusion_matrix_abmil_uni_v2.png
    └── ...
```

## CSV Output Format

The `experiment_results.csv` contains:
- `experiment_name` - Display name
- `model_config` - Full model configuration string
- `test_accuracy` - Test set accuracy
- `test_balanced_accuracy` - Balanced accuracy
- `test_quadratic_kappa` - Cohen's Kappa (quadratic weighted)
- `best_val_loss` - Best validation loss achieved
- `final_epoch` - Final epoch number
- `model_path` - Path to saved model weights
- `confusion_matrix_path` - Path to confusion matrix image

## Configuration Options

### Training Parameters

Edit these in the script:

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

### Data Parameters

```python
GRADE_GROUP = True
EXCLUDE_MID_GRADE = True
SEED = 10
TRAIN_SEED = 42
```

## Example: Running Multiple MIL Models

To compare different MIL architectures with the same encoder:

```python
EXPERIMENTS = [
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('transmil.base.uni_v2.pc108-24k', 'TransMIL + UNI_v2'),
    ('clam.base.uni_v2.pc108-24k', 'CLAM + UNI_v2'),
    ('dsmil.base.uni_v2.pc108-24k', 'DSMIL + UNI_v2'),
]

FEATURE_PATHS = {
    'uni_v2': '/path/to/features_uni_v2/',
}
```

## Example: Running Same Model with Different Encoders

```python
EXPERIMENTS = [
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('abmil.base.gigapath.pc108-24k', 'ABMIL + GigaPath'),
    ('abmil.base.conch_v15.pc108-24k', 'ABMIL + CONCH_v1.5'),
    ('abmil.base.virchow2.pc108-24k', 'ABMIL + Virchow2'),
]

FEATURE_PATHS = {
    'uni_v2': '/path/to/features_uni_v2/',
    'gigapath': '/path/to/features_gigapath/',
    'conch_v15': '/path/to/features_conch_v15/',
    'virchow2': '/path/to/features_virchow2/',
}
```

## Tips

1. **Check Feature Paths**: Ensure all feature directories exist before running
2. **GPU Memory**: Monitor GPU memory if running large models
3. **Early Stopping**: Adjust `EARLY_STOPPING_PATIENCE` based on your dataset
4. **Model Configs**: Check `src/model_configs/` for available model configurations
5. **Error Handling**: The script continues with next experiment if one fails

## Differences from Original run_mil.py

The new script:
- ✅ Runs multiple experiments in sequence
- ✅ Generates CSV with all metrics
- ✅ Creates confusion matrices for each run
- ✅ Supports all MIL models (not just ABMIL)
- ✅ Automatically maps encoders to feature paths
- ✅ Saves experiment configuration
- ✅ Creates timestamped output directories
- ✅ Handles errors gracefully (continues with next experiment)

## Original Script

The original `run_mil.py` is still available and unchanged for single-experiment runs.
