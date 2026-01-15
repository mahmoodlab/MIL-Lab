# Quick Start Guide - Visualization Directory

This is a quick reference for generating high-resolution heatmaps using JSON configuration.

## Three Steps to Generate Heatmaps

### 1. Copy and Edit Config

```bash
cp config_example.json my_heatmap.json
```

Edit `my_heatmap.json` with your paths:
```json
{
  "checkpoint": "/path/to/your/model.pt",
  "slide": "/path/to/your/slide.tiff",
  "output": "/path/to/output/"
}
```

**Note**: Output should be a directory path (ending with `/`). The filename will be auto-generated from the slide name and all config parameters.

### 2. Run the Script

```bash
python generate_heatmaps_highres.py --config my_heatmap.json
```

### 3. View the Output

Your heatmap will be saved with a descriptive filename containing all parameters.

**Example output**: `patient001_pc108-24k_uni_v2_patch256_overlap90_level0_jet_alpha50.tiff`

The filename includes:
- Slide name
- Model name (shortened)
- Feature extractor
- Patch size
- Overlap percentage
- Visualization level
- Colormap
- Alpha transparency

## Presets

Choose the right config for your needs:

| Config File | Quality | Speed | Best For |
|-------------|---------|-------|----------|
| `config_fast.json` | Preview | ~1-2 min | Quick debugging |
| `config_example.json` | Production | ~5-10 min | Regular use |
| `config_high_overlap.json` | Publication | ~15-30 min | Papers/presentations |

## Common Configurations

### Change Overlap (Smoothness)

```json
{
  "overlap": 0.90  // 0.70 = fast, 0.90 = standard, 0.95 = ultra-smooth
}
```

### Change Output Resolution

```json
{
  "vis_level": 0  // 0 = full res, 1 = 2x downsample, 2 = 4x downsample
}
```

### Change Colormap

```json
{
  "cmap": "jet"  // Options: "jet", "coolwarm", "viridis", "hot", "plasma"
}
```

### Disable Coordinate Saving

```json
{
  "save_coords": false  // Set to false to skip saving .h5 coord file
}
```

## Batch Processing Example

Process multiple slides with Python:

```python
import json
import subprocess
from pathlib import Path

# Base configuration
base_config = {
    "checkpoint": "models/my_model.pt",
    "model_name": "abmil.base.uni_v2.pc108-24k",
    "num_classes": 2,
    "overlap": 0.90,
}

# Process all slides in directory
slides_dir = Path("path/to/slides/")
output_dir = Path("path/to/heatmaps/")
output_dir.mkdir(exist_ok=True, parents=True)

for slide_path in slides_dir.glob("*.tiff"):
    # Create config for this slide
    config = base_config.copy()
    config["slide"] = str(slide_path)
    config["output"] = str(output_dir) + "/"  # Directory path - filename auto-generated

    # Save and run
    config_file = f"temp_{slide_path.stem}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Processing {slide_path.name}...")
    subprocess.run(["python", "generate_heatmaps_highres.py", "--config", config_file])
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size_extract` to 16 or 8
- Reduce `batch_size_infer` to 256 or 128

### Too Slow
- Use `vis_level: 1` (2x faster)
- Reduce `overlap` to 0.70 or 0.80

### Feature Mismatch Error
- Use `"feature_extractor": "auto"` (recommended)
- Or manually specify: `"feature_extractor": "uni_v2"` or `"uni_v1"`

## Full Documentation

See `README.md` in this directory for complete documentation.
