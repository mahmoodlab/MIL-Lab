# Heatmap Generation - Quick Start Guide

## Overview

This guide will help you generate attention heatmaps for your trained MIL models in **5 minutes**.

The heatmaps show which regions of a whole slide image your model is "paying attention to" when making predictions.

## Prerequisites

✅ Trained MIL model checkpoint (`.pt` file)
✅ Whole slide images (`.svs`, `.tiff`, etc.)
✅ Pre-extracted features (`.pt` files) and coordinates (`.h5` files)
✅ OpenSlide installed: `pip install openslide-python`

## 3-Step Process

### Step 1: Create Process List (1 minute)

Generate a CSV listing which slides to process:

```bash
python create_process_list.py \
    --slide_dir /path/to/your/slides \
    --features_dir /path/to/your/features \
    --output heatmaps/process_lists/my_slides.csv
```

**Example output:**
```
Found 50 slides with extension '.svs'
Found features for 48 slides
Saved process list to: heatmaps/process_lists/my_slides.csv
```

### Step 2: Configure Heatmap Generation (2 minutes)

Copy and edit the config template:

```bash
cp heatmaps/configs/config_template.yaml heatmaps/configs/my_experiment.yaml
```

**Edit these 3 essential fields** in `my_experiment.yaml`:

```yaml
model_arguments:
  ckpt_path: /path/to/your/checkpoint.pt    # ← YOUR MODEL

data_arguments:
  data_dir: /path/to/your/slides/           # ← YOUR SLIDES
  features_dir: /path/to/your/features/     # ← YOUR FEATURES
  process_list: heatmaps/process_lists/my_slides.csv
```

**Ensure model config matches training:**

```yaml
model_arguments:
  model_type: abmil      # ← YOUR MODEL TYPE (abmil, clam_sb, etc.)
  n_classes: 2           # ← YOUR NUMBER OF CLASSES
  in_dim: 1024           # ← YOUR FEATURE DIMENSION
```

### Step 3: Generate Heatmaps (2 minutes)

Run the generation script:

```bash
python create_heatmaps.py --config heatmaps/configs/my_experiment.yaml
```

**You'll see:**
```
Loading checkpoint from: /path/to/checkpoint.pt
Model loaded successfully!

Processing slides: 100%|████████████| 48/48 [02:15<00:00]

COMPLETE
Results saved to: heatmaps/results/MY_EXPERIMENT_results.csv
Heatmaps saved to: heatmaps/heatmap_production_results/
```

## View Your Heatmaps

### Option 1: Simple Image Viewer

Navigate to output directory and open images:

```bash
cd heatmaps/heatmap_production_results/MY_EXPERIMENT/tumor/
ls
# slide_001_heatmap.png
# slide_001_original.png
# slide_002_heatmap.png
# ...
```

### Option 2: Synchronized WSI Viewer

View original slide and heatmap side-by-side at the same location:

```bash
python view_heatmap_sync.py \
    --slide /path/to/slide.svs \
    --heatmap heatmaps/heatmap_production_results/MY_EXP/tumor/slide_001_heatmap.png
```

Use sliders and arrow buttons to navigate both images simultaneously.

## Customize Visualization

### Different Color Schemes

Edit config:

```yaml
heatmap_arguments:
  cmap: jet        # Default: blue → red
  # cmap: coolwarm  # Blue → white → red
  # cmap: viridis   # Purple → green → yellow
  # cmap: plasma    # Purple → pink → yellow
```

### Transparency

```yaml
heatmap_arguments:
  alpha: 0.4       # 0 = original only, 1 = heatmap only
```

### Resolution

```yaml
heatmap_arguments:
  vis_level: -1    # -1 = auto (~32x), 0 = full res (slow!)
```

## Common Issues

### "Features not found"

**Problem:** Script can't find feature files

**Solution:** Update `features_dir` in config to correct path

```yaml
data_arguments:
  features_dir: /correct/path/to/features/
```

### "Checkpoint incompatible"

**Problem:** Model config doesn't match checkpoint

**Solution:** Ensure `model_arguments` matches training config exactly:
- `model_type`: abmil, clam_sb, clam_mb, etc.
- `n_classes`: number of classes
- `in_dim`: feature dimension (1024 for UNI, 2048 for ResNet50)

### Out of Memory

**Problem:** Heatmap too large for GPU/RAM

**Solution:** Use lower resolution:

```yaml
heatmap_arguments:
  vis_level: 3            # Higher = lower resolution
  custom_downsample: 2    # Additional 2x downsampling
```

## What You Get

### 1. Heatmap Visualizations
**Location:** `heatmaps/heatmap_production_results/{exp}/{label}/`

- `{slide}_heatmap.png` - Attention overlay on original image
- `{slide}_original.png` - Original H&E image (if `save_orig: true`)

### 2. Raw Attention Data
**Location:** `heatmaps/heatmap_raw_results/{exp}/{label}/{slide}/`

- `{slide}_blockmap.h5` - HDF5 file with:
  - `attention_scores`: Attention values for each patch
  - `coords`: Patch coordinates (x, y) at level 0

### 3. Results Summary
**Location:** `heatmaps/results/{exp}_results.csv`

```csv
slide_id,label,bag_size,Pred_0,Pred_1,p_0,p_1
slide_001,tumor,1234,0,1,0.1234,0.8766
slide_002,normal,2345,0,1,0.9123,0.0877
```

## Next Steps

### 1. Analyze Results

Load results in Python:

```python
import pandas as pd
results = pd.read_csv('heatmaps/results/MY_EXP_results.csv')

# Check predictions
print(results[['slide_id', 'label', 'Pred_1', 'p_1']])

# Find high-confidence predictions
high_conf = results[results['p_1'] > 0.9]
```

### 2. Sample Top Patches

Add to config to save individual high-attention patches:

```yaml
sample_arguments:
  samples:
    - name: "top_patches"
      sample: true
      k: 20          # Save top 20 patches per slide
      mode: topk
```

Output: `heatmaps/heatmap_production_results/{exp}/sampled_patches/`

### 3. Further Analysis

Load raw attention data:

```python
import h5py
import numpy as np

with h5py.File('heatmaps/heatmap_raw_results/EXP/label/slide/slide_blockmap.h5', 'r') as f:
    attention = f['attention_scores'][:]
    coords = f['coords'][:]

# Find coordinates of highest attention
top_idx = np.argsort(attention.flatten())[-10:]  # Top 10
top_coords = coords[top_idx]
top_attention = attention[top_idx]

print(f"Top 10 patch coordinates: {top_coords}")
print(f"Top 10 attention scores: {top_attention}")
```

## Example Workflows

### Workflow 1: PANDA Dataset

```bash
# 1. Create process list
python create_process_list.py \
    --slide_dir /data/panda/train_images \
    --features_dir /data/panda/features/h5_files \
    --slide_ext .tiff \
    --output heatmaps/process_lists/panda_train.csv

# 2. Use PANDA-specific config
python create_heatmaps.py --config heatmaps/configs/config_abmil_panda.yaml
```

### Workflow 2: Multiple Experiments

```bash
# Compare different models
for model in abmil clam transmil; do
    python create_heatmaps.py \
        --config heatmaps/configs/config_${model}.yaml \
        --save_exp_code ${model}_comparison
done
```

### Workflow 3: Specific Slides Only

Create CSV: `selected_slides.csv`
```csv
slide_id,process
slide_001,1
slide_042,1
slide_099,1
```

Update config:
```yaml
data_arguments:
  process_list: selected_slides.csv
```

## Tips & Tricks

### Faster Processing
- Use `vis_level: 3` or higher for lower resolution
- Set `save_orig: false` to skip saving original images
- Process slides in batches by using multiple process lists

### Better Visualizations
- Try different colormaps: `jet`, `coolwarm`, `plasma`, `viridis`
- Adjust transparency: `alpha: 0.3` for subtle overlay, `alpha: 0.6` for prominent
- Enable blur for smoother appearance: `blur: true`

### High-Quality Outputs
- Use `save_ext: png` for lossless quality
- Set `vis_level: 1` for higher resolution
- Use `convert_to_percentiles: true` for better contrast

## Getting Help

- **Full documentation:** See [HEATMAP_GENERATION.md](HEATMAP_GENERATION.md)
- **Configuration help:** See [heatmaps/README.md](heatmaps/README.md)
- **Issues:** Check troubleshooting section in full docs

## Summary

You now know how to:
- ✅ Create a process list from your slides
- ✅ Configure heatmap generation
- ✅ Generate attention heatmaps
- ✅ View and analyze results
- ✅ Customize visualizations
- ✅ Extract top-attention patches

**Time to first heatmap: ~5 minutes** ⚡

Happy visualizing! 🔥
