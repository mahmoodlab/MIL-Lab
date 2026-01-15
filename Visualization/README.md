# Visualization Directory

This directory contains scripts for generating high-resolution attention heatmaps from trained MIL models.

## Contents

- `generate_heatmaps_highres.py` - Main script for generating high-resolution heatmaps with overlapping patches
- `config_example.json` - Example configuration file (standard quality, 90% overlap)
- `config_high_overlap.json` - High overlap configuration (95% overlap for ultra-smooth heatmaps)
- `config_fast.json` - Fast configuration (70% overlap for quick previews)

## Quick Start

### 1. Prepare Your Configuration

Copy one of the example config files and edit it with your paths:

```bash
cp config_example.json my_config.json
# Edit my_config.json with your paths
```

### 2. Run Heatmap Generation

```bash
python generate_heatmaps_highres.py --config my_config.json
```

## Configuration Options

All configuration is done via JSON files. Here are the available options:

### Required Fields

```json
{
  "checkpoint": "/path/to/model.pt",     // Path to trained model checkpoint
  "slide": "/path/to/slide.tiff",       // Path to whole slide image
  "output": "/path/to/output/"          // Output directory (auto-generates filename)
                                         // OR full path: "/path/to/output/custom.tiff"
}
```

**Smart Output Path Handling:**
- **Directory path** (e.g., `"output": "/path/to/heatmaps/"`): Automatically generates descriptive filename
  - Format: `{slide}_{model}_{encoder}_patch{size}_overlap{pct}_level{vis}_{cmap}_alpha{pct}.tiff`
  - Example: `patient001_pc108-24k_uni_v2_patch256_overlap90_level0_jet_alpha50.tiff`
  - Includes all relevant config parameters in the filename for easy identification
- **Full file path** (e.g., `"output": "/path/to/custom_name.tiff"`): Uses exact filename
- **No extension**: Treats as directory and auto-generates filename

### Optional Fields (with defaults)

```json
{
  "model_name": "abmil.base.uni_v2.pc108-24k",  // Model architecture identifier
  "num_classes": 2,                              // Number of output classes

  "feature_extractor": "auto",                   // 'auto', 'uni_v1', 'uni_v2', 'resnet50'

  "overlap": 0.90,                               // Overlap ratio: 0.90 = 90% overlap
  "patch_size": 256,                             // Patch size in pixels

  "vis_level": 0,                                // Pyramid level: 0=full res, 1=2x downsample

  "batch_size_extract": 32,                      // Batch size for feature extraction
  "batch_size_infer": 512,                       // Batch size for attention inference

  "cmap": "jet",                                 // Colormap: 'jet', 'coolwarm', 'viridis', etc.
  "alpha": 0.5,                                  // Transparency: 0.0-1.0

  "save_coords": true                            // Save coordinates and attention to .h5 file
}
```

## Configuration Presets

### Standard Quality (90% overlap)
- **Config**: `config_example.json`
- **Use case**: Production-quality heatmaps
- **Overlap**: 90%
- **Step size**: 26px (with 256px patches)
- **Processing time**: ~5-10 minutes per slide (GPU)

### Ultra High Quality (95% overlap)
- **Config**: `config_high_overlap.json`
- **Use case**: Publication-quality ultra-smooth heatmaps
- **Overlap**: 95%
- **Step size**: 13px (with 256px patches)
- **Processing time**: ~15-30 minutes per slide (GPU)
- **Note**: Much smoother gradients but 4x more patches to process

### Fast Preview (70% overlap)
- **Config**: `config_fast.json`
- **Use case**: Quick previews and debugging
- **Overlap**: 70%
- **Step size**: 77px (with 256px patches)
- **Vis level**: 1 (2x downsampled output)
- **Processing time**: ~1-2 minutes per slide (GPU)

## Feature Extractors

The script supports automatic feature extractor detection based on the model checkpoint. Supported extractors:

| Feature Extractor | Feature Dimension | Notes |
|-------------------|-------------------|-------|
| `uni_v2` (UNI2-h) | 1536 | Default, best quality |
| `uni_v1` (UNI) | 1024 | Original UNI model |
| `resnet50` | 1024 | Fast baseline |
| `ctranspath` | 768 | Not yet implemented - use Trident |
| `conch_v1` | 512 | Not yet implemented - use Trident |
| `virchow` | 2560 | Not yet implemented - use Trident |

Use `"feature_extractor": "auto"` to automatically detect from checkpoint (recommended).

## Output Files

The script generates:

1. **Heatmap image**: TIFF or PNG file with attention overlay
   - TIFF files use LZW compression
   - Recommended for high-resolution outputs

2. **Coordinates file** (optional): HDF5 file with patch coordinates and attention scores
   - Named as `<output>_coords.h5`
   - Contains:
     - `coords`: (N, 2) array of patch coordinates
     - `attention_scores`: (N, 1) array of attention values
   - Set `"save_coords": false` to disable

## Examples

### Example 1: Auto-Generated Filename (Recommended)

```json
{
  "checkpoint": "models/abmil_uni_v2_trained.pt",
  "slide": "slides/patient001.tiff",
  "output": "heatmaps/",

  "model_name": "abmil.base.uni_v2.pc108-24k",
  "num_classes": 2,
  "feature_extractor": "auto",

  "overlap": 0.90,
  "patch_size": 256,
  "vis_level": 0
}
```
**Output**: `heatmaps/patient001_pc108-24k_uni_v2_patch256_overlap90_level0_jet_alpha50.tiff`

The auto-generated filename includes all parameters for easy identification.

### Example 2: Custom Filename

```json
{
  "checkpoint": "models/abmil_uni_v2_trained.pt",
  "slide": "slides/patient001.tiff",
  "output": "heatmaps/custom_name_for_paper.tiff",

  "overlap": 0.90,
  "patch_size": 256,
  "vis_level": 0
}
```
**Output**: `heatmaps/custom_name_for_paper.tiff`

### Example 3: Fast Preview with Auto-Generated Name

```json
{
  "checkpoint": "models/abmil_uni_v2_trained.pt",
  "slide": "slides/patient001.tiff",
  "output": "heatmaps/previews/",

  "overlap": 0.70,
  "vis_level": 1,
  "save_coords": false
}
```
**Output**: `heatmaps/previews/patient001_pc108-24k_auto_patch256_overlap70_level1_jet_alpha50.tiff`

### Example 4: Publication Quality with Different Colormap

```json
{
  "checkpoint": "models/transmil_trained.pt",
  "slide": "slides/patient001.tiff",
  "output": "heatmaps/publication/",

  "model_name": "transmil.base.uni_v2.pc108-24k",
  "overlap": 0.95,
  "cmap": "coolwarm",
  "alpha": 0.6
}
```
**Output**: `heatmaps/publication/patient001_pc108-24k_uni_v2_patch256_overlap95_level0_coolwarm_alpha60.tiff`

## Understanding Overlap

The overlap parameter controls how densely patches are sampled:

- **Overlap = 0.90** (90%):
  - Step size = 26px (with 256px patches)
  - Each point in the image is covered by ~100 patches
  - Smooth gradients, good balance of quality and speed

- **Overlap = 0.95** (95%):
  - Step size = 13px (with 256px patches)
  - Each point in the image is covered by ~400 patches
  - Ultra-smooth gradients, but 4x more computation

- **Overlap = 0.70** (70%):
  - Step size = 77px (with 256px patches)
  - Each point in the image is covered by ~11 patches
  - Faster but less smooth, good for previews

## Performance Tips

1. **GPU Memory**: Use smaller batch sizes if you run out of GPU memory
   - Reduce `batch_size_extract` (e.g., 16 instead of 32)
   - Reduce `batch_size_infer` (e.g., 256 instead of 512)

2. **Speed**: For faster generation:
   - Use `vis_level: 1` or `2` for downsampled output
   - Reduce `overlap` (e.g., 0.70 or 0.80)
   - Increase batch sizes if you have GPU memory

3. **Quality**: For best quality:
   - Use `overlap: 0.95` or higher
   - Use `vis_level: 0` for full resolution
   - Save as TIFF for lossless compression

4. **Disk Space**:
   - TIFF files are larger but higher quality
   - PNG files are smaller but may have compression artifacts
   - Set `save_coords: false` if you don't need coordinate data

## Troubleshooting

### Out of GPU Memory
- Reduce `batch_size_extract` to 16 or 8
- Reduce `batch_size_infer` to 256 or 128
- Use smaller overlap (e.g., 0.80)

### Slow Performance
- Check GPU utilization: `nvidia-smi`
- Increase batch sizes if GPU is underutilized
- Use `vis_level: 1` for 2x faster generation

### Feature Extractor Mismatch
- Verify checkpoint was trained with correct encoder
- Use `"feature_extractor": "auto"` for automatic detection
- Check model architecture matches `model_name`

### File Not Found Errors
- Use absolute paths in config file
- Check file permissions
- Verify checkpoint file is valid PyTorch model

## Advanced Usage

### Batch Processing Multiple Slides

Create a simple Python script to process multiple slides:

```python
import json
import subprocess
from pathlib import Path

# Base config
base_config = {
    "checkpoint": "models/abmil_trained.pt",
    "model_name": "abmil.base.uni_v2.pc108-24k",
    "num_classes": 2,
    "overlap": 0.90,
}

# Process multiple slides
slides_dir = Path("slides/")
output_dir = Path("heatmaps/")
output_dir.mkdir(exist_ok=True)

for slide_path in slides_dir.glob("*.tiff"):
    # Create config for this slide
    config = base_config.copy()
    config["slide"] = str(slide_path)
    config["output"] = str(output_dir / f"{slide_path.stem}_heatmap.tiff")

    # Save config
    config_path = f"temp_config_{slide_path.stem}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run heatmap generation
    print(f"Processing {slide_path.name}...")
    subprocess.run([
        "python", "generate_heatmaps_highres.py",
        "--config", config_path
    ])
```

### Custom Region of Interest (ROI)

To process only a specific region, you would need to modify the config to include ROI coordinates. This is not currently supported but could be added.

## Citation

If you use this heatmap generation code in your research, please cite:

```bibtex
@software{millab_visualization,
  title={MIL-Lab Visualization Tools},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/MIL-Lab}
}
```

## License

This code is part of the MIL-Lab project. See the main repository for license information.

---

**Last Updated**: 2025-12-05
