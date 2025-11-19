# Heatmap Generation Quick Reference

## Quick Start

```bash
# 1. Create config from template
cp heatmaps/configs/config_template.yaml heatmaps/configs/my_experiment.yaml

# 2. Edit config - update these paths:
#    - model_arguments.ckpt_path: your model checkpoint
#    - data_arguments.data_dir: your WSI directory
#    - data_arguments.features_dir: your features directory

# 3. Run heatmap generation
python create_heatmaps.py --config heatmaps/configs/my_experiment.yaml

# 4. View results
python view_heatmap_sync.py \
    --slide /path/to/slide.svs \
    --heatmap heatmaps/heatmap_production_results/MY_EXP/label/slide_heatmap.png
```

## Directory Structure

```
heatmaps/
├── configs/                           # Configuration files
│   ├── config_template.yaml          # Generic template
│   └── config_abmil_panda.yaml       # PANDA-specific example
├── process_lists/                     # CSV files with slide lists
│   └── example_process_list.csv
├── heatmap_raw_results/              # Raw attention scores & coords
│   └── {experiment_name}/
│       └── {label}/
│           └── {slide_id}/
│               └── {slide_id}_blockmap.h5
├── heatmap_production_results/       # Final visualizations
│   └── {experiment_name}/
│       └── {label}/
│           ├── {slide_id}_heatmap.png
│           ├── {slide_id}_original.png
│           └── sampled_patches/
└── results/                           # CSV results summaries
    └── {experiment_name}_results.csv
```

## Essential Config Parameters

### Paths (MUST UPDATE)
```yaml
model_arguments:
  ckpt_path: /path/to/your/checkpoint.pt  # Your trained model

data_arguments:
  data_dir: /path/to/slides/              # Your WSI files
  features_dir: /path/to/features/        # Your extracted features
```

### Model Architecture (MUST MATCH TRAINING)
```yaml
model_arguments:
  model_type: abmil     # abmil, clam_sb, clam_mb, transmil, dsmil
  n_classes: 2          # Number of classes
  in_dim: 1024          # Feature dimension (UNI=1024, ResNet=2048)
```

### Visualization Style
```yaml
heatmap_arguments:
  cmap: jet             # Color scheme
  alpha: 0.4            # Transparency (0-1)
  vis_level: -1         # Resolution (-1=auto)
```

## Common Use Cases

### 1. Process All Slides in Directory
```yaml
data_arguments:
  data_dir: /path/to/slides/
  process_list: null    # null = process all
```

### 2. Process Specific Slides
Create CSV: `heatmaps/process_lists/my_slides.csv`
```csv
slide_id,label,process
slide_001,tumor,1
slide_002,normal,1
slide_003,tumor,0
```

Config:
```yaml
data_arguments:
  process_list: heatmaps/process_lists/my_slides.csv
```

### 3. Different Colormaps
```yaml
heatmap_arguments:
  cmap: jet        # Blue -> Red (classic)
  # cmap: coolwarm  # Blue -> White -> Red
  # cmap: viridis   # Purple -> Green -> Yellow
  # cmap: plasma    # Purple -> Pink -> Yellow
```

### 4. High-Resolution Output
```yaml
heatmap_arguments:
  vis_level: 0           # Full resolution (slow!)
  save_ext: tiff         # Lossless format
  custom_downsample: 1   # No additional downsampling
```

### 5. Fast Processing (Lower Resolution)
```yaml
heatmap_arguments:
  vis_level: 3           # Higher level = lower res
  save_ext: jpg          # Smaller files
  custom_downsample: 2   # Additional 2x downsample
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Features not found" | Update `features_dir` in config |
| "Checkpoint incompatible" | Ensure `model_arguments` matches training config |
| "OpenSlide error" | Install: `pip install openslide-python` |
| Out of memory | Increase `vis_level` or `custom_downsample` |
| No attention scores | Check model has `return_attention=True` |

## Output Files

### Heatmap Image
- **Location**: `heatmap_production_results/{exp}/{label}/{slide}_heatmap.png`
- **Content**: Attention overlay on original slide
- **Format**: PNG/JPG/TIFF
- **Use**: Visual inspection, presentations

### Raw Attention Data
- **Location**: `heatmap_raw_results/{exp}/{label}/{slide}/{slide}_blockmap.h5`
- **Content**: HDF5 with `attention_scores` and `coords` arrays
- **Use**: Further analysis, custom visualizations

### Results Summary
- **Location**: `heatmap/results/{exp}_results.csv`
- **Content**: Predictions, probabilities, bag sizes
- **Use**: Statistical analysis, validation

## Examples

### Example 1: PANDA Dataset
```bash
python create_heatmaps.py --config heatmaps/configs/config_abmil_panda.yaml
```

### Example 2: Custom Colormap
Edit config:
```yaml
heatmap_arguments:
  cmap: plasma
  alpha: 0.5
```

### Example 3: Sample Top Patches
Edit config:
```yaml
sample_arguments:
  samples:
    - name: "high_attention"
      sample: true
      k: 20
      mode: topk
```

Output: `heatmap_production_results/{exp}/sampled_patches/label_X_pred_Y/high_attention/`

## See Also

- [Full Documentation](../HEATMAP_GENERATION.md)
- [Main README](../README.md)
- [OpenSlide Documentation](https://openslide.org/api/python/)
