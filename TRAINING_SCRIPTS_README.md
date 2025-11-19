# MIL Training Scripts

Refactored training and visualization scripts with shared utilities for both CLAM and ABMIL models.

## 📁 Structure

```
MIL-Lab/
├── utils/
│   ├── __init__.py          # Package exports
│   ├── data_utils.py        # Data loading and preprocessing
│   └── vis_utils.py         # Visualization utilities
├── train_abmil_simple.py    # Simplified ABMIL training script
├── train_clam_simple.py     # Simplified CLAM training script
├── visualize_model.py       # Standalone visualization script
├── train_abmil_feather_panda.py  # Full ABMIL script (legacy)
└── train_clam_panda.py      # Full CLAM script (legacy)
```

## ✨ Key Features

### Shared Utilities (`utils/`)

#### Data Utils (`data_utils.py`)
- ✅ **Format-agnostic dataset**: Handles both 2D (new Trident) and 3D (old) embeddings
- ✅ **Unified preprocessing**: Single function for data preparation
- ✅ **Universal dataset class**: Works for both CLAM and ABMIL
- ✅ **Automatic format detection**: No manual configuration needed

#### Visualization Utils (`vis_utils.py`)
- ✅ **Model-agnostic**: Works with both CLAM and ABMIL
- ✅ **Top patches extraction**: Visualize highest attention patches
- ✅ **Heatmap generation**: Overlay attention on WSIs
- ✅ **Confusion matrix**: Automated plotting

## 🚀 Usage

### Quick Start: Simplified Scripts

#### Train ABMIL with FEATHER-24K
```bash
python train_abmil_simple.py
```

**Key parameters** (edit script):
- `BATCH_SIZE = 32` (can use >1 for ABMIL)
- `NUM_FEATURES = 512` (sample 512 patches)
- Pretrained on PC-108 (FEATHER-24K)

#### Train CLAM
```bash
python train_clam_simple.py
```

**Key parameters** (edit script):
- `BATCH_SIZE = 1` (required for CLAM)
- `NUM_FEATURES = None` (use all patches)
- `K_SAMPLE = 8` (top-K instance sampling)
- `BAG_WEIGHT = 0.7` (dual loss weighting)

### Visualization

After training, visualize predictions:

```bash
# Visualize ABMIL model
python visualize_model.py \
  --model-type abmil \
  --model-path best_model_abmil.pth \
  --csv-path /path/to/train.csv \
  --feats-path /path/to/features/ \
  --wsi-dir /path/to/slides/ \
  --output-dir ./viz_abmil \
  --num-classes 3 \
  --grade-group \
  --exclude-mid-grade \
  --top-k 3

# Visualize CLAM model
python visualize_model.py \
  --model-type clam \
  --model-path best_model_clam.pth \
  --csv-path /path/to/train.csv \
  --feats-path /path/to/features/ \
  --wsi-dir /path/to/slides/ \
  --output-dir ./viz_clam \
  --num-classes 3 \
  --grade-group \
  --exclude-mid-grade
```

**Visualization options:**
- `--skip-patches`: Skip top patches visualization
- `--skip-heatmaps`: Skip heatmap visualization
- `--top-k N`: Number of top patches to extract (default: 3)

## 🔧 Configuration

### Data Paths (edit in scripts)

```python
CSV_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
FEATS_PATH = '/media/nadim/Data/.../features_uni_v2/'
WSI_DIR = '/media/nadim/Data/.../train_images'
```

### Grade Grouping

```python
GRADE_GROUP = True           # Use clinical grade grouping
EXCLUDE_MID_GRADE = True     # Exclude ISUP 2-3 (3-class problem)
```

Options:
1. **3-class** (recommended): `GRADE_GROUP=True, EXCLUDE_MID_GRADE=True`
   - Group 0: No cancer (ISUP 0)
   - Group 1: Low grade (ISUP 1)
   - Group 2: High grade (ISUP 4-5)

2. **4-class**: `GRADE_GROUP=True, EXCLUDE_MID_GRADE=False`
   - Group 0: No cancer (ISUP 0)
   - Group 1: Low grade (ISUP 1)
   - Group 2: Mid grade (ISUP 2-3)
   - Group 3: High grade (ISUP 4-5)

3. **6-class** (original): `GRADE_GROUP=False`
   - ISUP 0, 1, 2, 3, 4, 5

## 📊 Model Comparison

| Feature | ABMIL (Simplified) | CLAM (Simplified) |
|---------|-------------------|------------------|
| **Script** | `train_abmil_simple.py` | `train_clam_simple.py` |
| **Batch size** | 32 | 1 (required) |
| **Patch sampling** | 512 patches | All patches |
| **Loss type** | Single (bag-level) | Dual (bag + instance) |
| **Pretrained** | Yes (FEATHER-24K) | No |
| **Training time** | Faster (sampled patches) | Slower (all patches) |
| **Memory** | Lower | Higher |

## 🐛 Troubleshooting

### "RuntimeError: batch1 must be a 3D tensor"
**Cause**: Using old code with new 2D Trident embeddings

**Solution**: Use the new scripts - they auto-detect 2D/3D formats!

### "RuntimeError: Trying to resize storage"
**Cause**: Missing `.clone()` for batch collation

**Solution**: The new `PANDAH5Dataset` handles this automatically

### Jupyter notebook not working
**Cause**: Kernel caching old class definitions

**Solution**:
1. Restart kernel
2. Re-run cells in order
3. Or use standalone scripts (recommended)

## 📝 Import Example

You can also import the utilities in your own scripts:

```python
from utils import preprocess_panda_data, create_dataloaders
from utils import visualize_top_patches, visualize_heatmaps, plot_confusion_matrix

# Preprocess data
df, num_classes, class_labels = preprocess_panda_data(
    csv_path, feats_path,
    grade_group=True,
    exclude_mid_grade=True,
    seed=10
)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    feats_path, df,
    batch_size=32,        # 32 for ABMIL, 1 for CLAM
    num_features=512,     # 512 for ABMIL, None for CLAM
    num_workers=4,
    seed=42
)

# Visualize after training
visualize_top_patches(
    model, df, feats_path, wsi_dir,
    num_classes, class_labels,
    criterion, device,
    model_type='abmil',
    output_dir='./viz'
)
```

## 🎯 Next Steps

1. **Run simplified training scripts** to verify they work with your data
2. **Use visualization script** to analyze model predictions
3. **Customize hyperparameters** by editing configuration section
4. **Integrate into your workflow** using the shared utilities

## 📚 Legacy Scripts

The full standalone scripts are still available:
- `train_abmil_feather_panda.py` - Full ABMIL with all features
- `train_clam_panda.py` - Full CLAM with all features

These include inline visualization code but are less modular than the new simplified versions.

## 🆘 Getting Help

If you encounter issues:
1. Check that paths are correct in configuration
2. Verify h5 files exist and are readable
3. Ensure embeddings are in correct format (auto-detected)
4. Check GPU memory for batch size issues

For the 2D/3D embedding format issue:
- **NEW Trident embeddings**: `(num_patches, 1536)` ✓ Supported
- **OLD embeddings**: `(1, num_patches, 1536)` ✓ Supported
- **Auto-detection**: The dataset automatically handles both formats
