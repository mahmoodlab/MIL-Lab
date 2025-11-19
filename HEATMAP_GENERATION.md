# Heatmap Generation for MIL-Lab Models

This guide explains how to generate attention heatmaps for whole slide images (WSIs) using trained MIL models. The heatmaps are compatible with OpenSlide and can be viewed synchronously with the original slides.

## Overview

The heatmap generation pipeline:
1. Loads a trained MIL model from checkpoint
2. Extracts attention scores for each patch in a slide
3. Creates a heatmap visualization overlaid on the original WSI
4. Saves results in OpenSlide-compatible format
5. Optionally samples top-k patches with highest attention

## Quick Start

### 1. Prepare Configuration File

Create a YAML configuration file (or use the template at `heatmaps/configs/config_template.yaml`):

```bash
cp heatmaps/configs/config_template.yaml heatmaps/configs/my_config.yaml
```

Edit `my_config.yaml` to specify:
- Path to your trained model checkpoint
- Directory containing WSI files
- Directory containing pre-extracted features
- Visualization parameters (colormap, transparency, etc.)

### 2. Run Heatmap Generation

```bash
python create_heatmaps.py --config heatmaps/configs/my_config.yaml
```

### 3. View Results

Heatmaps will be saved to the directory specified in `production_save_dir`:
```
heatmaps/heatmap_production_results/
  └── {save_exp_code}/
      └── {label}/
          ├── slide_001_heatmap.png
          ├── slide_001_original.png
          ├── slide_002_heatmap.png
          └── ...
```

Raw attention scores and coordinates are saved to `raw_save_dir`:
```
heatmaps/heatmap_raw_results/
  └── {save_exp_code}/
      └── {label}/
          └── slide_001/
              ├── slide_001_blockmap.h5  # Attention scores + coordinates
              └── ...
```

## Configuration Guide

### Required Sections

#### `exp_arguments`
```yaml
exp_arguments:
  n_classes: 2                                      # Number of output classes
  save_exp_code: MY_EXPERIMENT                      # Experiment identifier
  raw_save_dir: heatmaps/heatmap_raw_results       # Raw data output
  production_save_dir: heatmaps/heatmap_production_results  # Final visualizations
  batch_size: 256                                   # Batch size for inference
```

#### `data_arguments`
```yaml
data_arguments:
  data_dir: /path/to/slides/                       # WSI directory
  slide_ext: .svs                                   # Slide file extension
  features_dir: features/h5_files                   # Pre-extracted features
  process_list: null                                # Optional: CSV with slide list
```

**Process List CSV Format** (optional):
```csv
slide_id,label,process,features_path,coords_path
slide_001,tumor,1,features/h5_files/slide_001.pt,features/h5_files/slide_001.h5
slide_002,normal,1,features/h5_files/slide_002.pt,features/h5_files/slide_002.h5
```
- `slide_id`: Slide identifier (with or without extension)
- `label`: Class label (optional)
- `process`: 1 = process this slide, 0 = skip
- `features_path`: Path to feature file (optional, defaults to `{features_dir}/{slide_id}.pt`)
- `coords_path`: Path to coordinates (optional, defaults to `{features_dir}/{slide_id}.h5`)

#### `model_arguments`
```yaml
model_arguments:
  ckpt_path: /path/to/checkpoint.pt                # Model checkpoint
  model_type: abmil                                 # Model architecture
  n_classes: 2
  in_dim: 1024                                      # Feature dimension
  attention_dim: 256                                # Attention dimension
  dropout: 0.25
```

Supported model types:
- `abmil`: Attention-based MIL
- `clam_sb`: CLAM single-branch
- `clam_mb`: CLAM multi-branch
- `transmil`: TransMIL
- `dsmil`: DS-MIL

#### `heatmap_arguments`
```yaml
heatmap_arguments:
  vis_level: -1                    # Visualization level (-1 = auto)
  alpha: 0.4                       # Transparency (0-1)
  cmap: jet                        # Colormap
  convert_to_percentiles: true     # Better contrast
  blur: false                      # Gaussian smoothing
  save_orig: true                  # Save original image
  save_ext: png                    # Output format (png/jpg/tiff)
```

**Colormap Options**:
- `jet`: Blue (low) → Red (high)
- `coolwarm`: Blue (low) → White → Red (high)
- `viridis`: Purple (low) → Green → Yellow (high)
- `plasma`: Purple (low) → Pink → Yellow (high)
- `hot`: Black (low) → Red → Yellow → White (high)

## Advanced Usage

### Custom Feature Paths

If your features are stored in non-standard locations, create a process list CSV:

```csv
slide_id,features_path,coords_path
slide_001,/custom/path/features_001.pt,/custom/path/coords_001.h5
slide_002,/custom/path/features_002.pt,/custom/path/coords_002.h5
```

Then reference it in your config:
```yaml
data_arguments:
  process_list: heatmaps/process_lists/my_custom_list.csv
```

### Multiple Data Directories

If slides are stored across multiple directories:

```yaml
data_arguments:
  data_dir:
    cohort_a: /path/to/cohort_a/slides/
    cohort_b: /path/to/cohort_b/slides/
  data_dir_key: source  # Column name in process list
```

Process list:
```csv
slide_id,source,label
slide_001,cohort_a,tumor
slide_002,cohort_b,normal
```

### Sampling Top Patches

Extract and save patches with highest attention scores:

```yaml
sample_arguments:
  samples:
    - name: "high_attention"
      sample: true
      k: 15           # Top 15 patches
      mode: topk
      seed: 1

    - name: "low_attention"
      sample: true
      k: 10           # Bottom 10 patches
      mode: bottomk
      seed: 1
```

Patches are saved to:
```
heatmaps/heatmap_production_results/{exp_code}/sampled_patches/
  └── label_{label}_pred_{prediction}/
      ├── high_attention/
      │   ├── 0_slide_001_x_1234_y_5678_score_0.9234.png
      │   └── ...
      └── low_attention/
          └── ...
```

### Region of Interest (ROI) Processing

To generate heatmaps for specific regions only:

```csv
slide_id,label,x1,y1,x2,y2
slide_001,tumor,1000,2000,5000,6000
```

The coordinates (x1, y1, x2, y2) define the bounding box at level 0.

## Viewing Heatmaps with OpenSlide

### Synchronized Viewing

To view the original slide and heatmap side-by-side at the exact same location, use the OpenSlide viewer examples:

```bash
cd ~/Source/openslide-python/examples

# View original slide
python deepzoom_multiserver.py --slides /path/to/slide.svs

# In another terminal, view heatmap
python deepzoom_multiserver.py --slides /path/to/heatmap.png
```

Then open both in your browser and navigate to the same region.

### Alternative: Create Synchronized Viewer Script

We provide a helper script for synchronized viewing (see next section).

## Helper Scripts

### Create Process List from Directory

Automatically create a process list from a slide directory:

```python
import os
import pandas as pd

slide_dir = "/path/to/slides"
features_dir = "features/h5_files"
slide_ext = ".svs"

slides = [f for f in os.listdir(slide_dir) if f.endswith(slide_ext)]
slide_ids = [f.replace(slide_ext, "") for f in slides]

df = pd.DataFrame({
    'slide_id': slide_ids,
    'process': 1,
    'features_path': [f'{features_dir}/{sid}.pt' for sid in slide_ids],
    'coords_path': [f'{features_dir}/{sid}.h5' for sid in slide_ids]
})

df.to_csv('heatmaps/process_lists/auto_generated.csv', index=False)
print(f"Created process list with {len(df)} slides")
```

### Batch Process Multiple Experiments

Process multiple configurations in sequence:

```bash
#!/bin/bash

configs=(
    "heatmaps/configs/experiment_1.yaml"
    "heatmaps/configs/experiment_2.yaml"
    "heatmaps/configs/experiment_3.yaml"
)

for config in "${configs[@]}"; do
    echo "Processing: $config"
    python create_heatmaps.py --config "$config"
done
```

## Troubleshooting

### Common Issues

**1. "Features not found"**
- Ensure `features_dir` points to the correct location
- Check that feature files are named `{slide_id}.pt`
- Use absolute paths if relative paths aren't working

**2. "Coordinates not found"**
- Coordinates should be in HDF5 files: `{slide_id}.h5`
- The H5 file should contain a `coords` dataset
- Verify with: `h5dump -n features.h5`

**3. "Model checkpoint incompatible"**
- Ensure `model_arguments` matches your training config
- Check that `n_classes`, `in_dim`, and architecture match
- Try loading checkpoint manually to debug

**4. "OpenSlide error"**
- Install OpenSlide: `sudo apt-get install openslide-tools`
- Install Python bindings: `pip install openslide-python`
- Verify slide format is supported: `.svs`, `.tif`, `.ndpi`, etc.

**5. "Out of memory"**
- Reduce `vis_level` (use higher pyramid level)
- Increase `custom_downsample`
- Process slides individually instead of batch

### Debug Mode

Add verbose logging by modifying the script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Attention Scores

Verify attention scores are being computed:

```python
import h5py
import numpy as np

with h5py.File('heatmaps/heatmap_raw_results/EXP/label/slide/slide_blockmap.h5', 'r') as f:
    scores = f['attention_scores'][:]
    coords = f['coords'][:]

print(f"Num patches: {len(scores)}")
print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
print(f"Coord range: [{coords.min()}, {coords.max()}]")
```

## Output Format

### Heatmap Images

- **Format**: PNG, JPG, or TIFF
- **Resolution**: Matches the visualization level (typically ~32x downsampled)
- **Colormap**: Applied to normalized attention scores
- **Transparency**: Blended with original slide image

### Raw Data (HDF5)

```python
# Structure of {slide_id}_blockmap.h5
{
    'attention_scores': np.ndarray,  # Shape: (N, 1), dtype: float32
    'coords': np.ndarray             # Shape: (N, 2), dtype: int32
}
```

Coordinates are at pyramid level 0 (full resolution).

### Results CSV

```csv
slide_id,label,bag_size,Pred_0,Pred_1,p_0,p_1
slide_001,tumor,1234,0,1,0.1234,0.8766
slide_002,normal,2345,0,1,0.9123,0.0877
```

- `bag_size`: Number of patches in the slide
- `Pred_X`: Predicted class (if matches Y_hat)
- `p_X`: Probability for class X

## Integration with Existing Workflows

### After Feature Extraction

```bash
# 1. Extract features (as usual)
python extract_features_fp.py --data_h5_dir ... --feat_dir features/h5_files

# 2. Train model
python train_abmil.py --config configs/my_config.yaml

# 3. Generate heatmaps
python create_heatmaps.py --config heatmaps/configs/my_heatmap_config.yaml
```

### With CLAM Workflow

The heatmap generation is compatible with CLAM's output format, so you can:
1. Use CLAM for preprocessing and feature extraction
2. Train with MIL-Lab models
3. Generate heatmaps with this script
4. View with CLAM's visualization tools

## Citation

If you use this heatmap generation tool, please cite:

```bibtex
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Additional Resources

- [OpenSlide Documentation](https://openslide.org/api/python/)
- [CLAM Repository](https://github.com/mahmoodlab/CLAM)
- [MIL-Lab Documentation](./README.md)
- [Matplotlib Colormaps](https://matplotlib.org/stable/tutorials/colors/colormaps.html)
