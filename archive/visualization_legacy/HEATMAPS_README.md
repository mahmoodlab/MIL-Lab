# Heatmap Generation - Simple Guide

Generate attention heatmaps from your trained MIL models in 1 command.

## Quick Start

```bash
python generate_heatmaps.py \
    --checkpoint path/to/model.pt \
    --h5_dir path/to/embeddings/ \
    --slide_dir path/to/slides/ \
    --output_dir heatmaps/output/
```

That's it! The script will:
1. Load your trained model
2. Find all H5 embedding files
3. Generate attention heatmaps for each slide
4. Save visualizations and results CSV

## Example (PANDA Dataset)

```bash
python generate_heatmaps.py \
    --checkpoint checkpoints/abmil_panda_best.pt \
    --h5_dir /data/panda/trident_processed/features_uni_v2/ \
    --slide_dir /data/panda/train_images/ \
    --output_dir heatmaps/panda_results/ \
    --slide_ext .tiff
```

## What You Get

```
heatmaps/output/
├── slide_001_heatmap.png       # Heatmap visualization
├── slide_002_heatmap.png
├── ...
└── heatmap_results.csv         # Predictions & attention stats
```

The CSV contains:
- `slide_id`: Slide identifier
- `num_patches`: Number of patches processed
- `prediction`: Model prediction
- `probabilities`: Class probabilities
- `attention_mean`: Average attention score
- `attention_std`: Attention score std dev
- `heatmap_path`: Path to heatmap image

## Options

### Visualization Style

```bash
# Different colormap
--cmap plasma          # Options: jet, coolwarm, viridis, plasma, hot

# More/less transparency
--alpha 0.5            # 0=original only, 1=heatmap only (default: 0.4)

# Different resolution
--vis_level 2          # Higher=lower res, faster (default: -1=auto)
```

### File Selection

```bash
# Process specific slides from CSV (must have 'slide_id' column)
--csv test_slides.csv

# Process only first N slides (for testing)
--limit 10

# Different slide extension
--slide_ext .svs       # Default: .tiff
```

### Model Configuration

```bash
# Different model
--model_name abmil.base.uni_v2.pc108-24k

# Number of classes
--num_classes 6        # Default: 2
```

## View Heatmaps

### Simple viewer (interactive navigation)

```bash
python view_heatmap_sync.py \
    --slide /path/to/slide.tiff \
    --heatmap heatmaps/output/slide_001_heatmap.png
```

### Or just open the PNG files directly

The heatmaps are standard PNG images - open them with any image viewer!

## How It Works

The script is designed for your H5 embedding format:

```python
# Your H5 file structure:
with h5py.File('slide.h5', 'r') as f:
    features = f['features'][:]  # (num_patches, 1536) - 2D format
    coords = f['coords'][:]       # (num_patches, 2)
```

The script:
1. Loads features and coordinates from H5 file
2. Runs your model to get attention scores
3. Maps attention scores to patch coordinates
4. Creates heatmap visualization on the original slide
5. Saves both heatmap and results

## Complete Example

```bash
# 1. Train your model (you've already done this)
python train_abmil_simple.py

# 2. Generate heatmaps for all slides
python generate_heatmaps.py \
    --checkpoint checkpoints/best_model.pt \
    --h5_dir /data/embeddings/ \
    --slide_dir /data/slides/ \
    --output_dir heatmaps/results/

# 3. View a specific slide
python view_heatmap_sync.py \
    --slide /data/slides/interesting_slide.tiff \
    --heatmap heatmaps/results/interesting_slide_heatmap.png

# 4. Analyze results
python
>>> import pandas as pd
>>> df = pd.read_csv('heatmaps/results/heatmap_results.csv')
>>> df[['slide_id', 'prediction', 'attention_mean']].head()
```

## Colormap Examples

Different colormaps highlight attention differently:

- **`jet`** (default): Blue (low) → Red (high), classic, high contrast
- **`coolwarm`**: Blue → White → Red, diverging, easier on eyes
- **`viridis`**: Purple → Green → Yellow, perceptually uniform
- **`plasma`**: Purple → Pink → Yellow, perceptually uniform, vibrant
- **`hot`**: Black → Red → Yellow → White, thermal-style

Try a few to see what works best for your data!

## Troubleshooting

**"Model name not found"**
- Update `--model_name` to match your trained model
- Or the script will try to load checkpoint directly (fallback)

**"H5 not found"**
- Check that `--h5_dir` path is correct
- Ensure H5 files are named `{slide_id}.h5`

**"Slide not found"**
- Check that `--slide_dir` path is correct
- Update `--slide_ext` to match your slide files (`.tiff`, `.svs`, etc.)

**Out of memory**
- Use lower resolution: `--vis_level 3`
- Process fewer slides at a time: `--limit 100`

## Files

- **`generate_heatmaps.py`** - Main script (clean, 300 lines)
- **`view_heatmap_sync.py`** - Interactive viewer
- **`example_generate_heatmaps.sh`** - Usage examples
- **`src/visualization/heatmap_utils.py`** - Visualization utilities

## Dependencies

All already installed from your MIL-Lab setup:
- torch
- h5py
- openslide-python
- matplotlib
- scipy
- PIL

Test with: `python test_heatmap_setup.py`

---

**That's it!** No complex configs, no YAML files, just one command to generate heatmaps.
