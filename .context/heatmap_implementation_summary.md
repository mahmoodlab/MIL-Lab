# Heatmap Visualization Implementation - Context Summary

**Date:** 2024-11-17
**Task:** Add attention heatmap visualization to MIL-Lab (CLAM-inspired, optimized for new workflow)

## What Was Built

Added complete heatmap generation system with TWO implementations:
1. **Simple version** (recommended) - Lean, command-line driven, optimized for current H5 format
2. **CLAM-compatible version** - Full-featured with YAML configs for backward compatibility

## Current Embedding Format (Important!)

The project uses **clean 2D H5 format** from Trident processing:

```python
# H5 file structure: slide_id.h5
with h5py.File('slide_001.h5', 'r') as f:
    features = f['features'][:]  # Shape: (num_patches, 1536) - 2D array
    coords = f['coords'][:]       # Shape: (num_patches, 2) - patch coordinates at level 0
```

**Key points:**
- Features are **2D** (num_patches, feature_dim), NOT 3D
- Standard feature extractors: UNI v2 (1536-dim), stored at 20x magnification, 256px patches
- Coordinates are at pyramid level 0 (full resolution)
- Files stored at: `/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/`

## Files Created

### RECOMMENDED: Simple Implementation

1. **`generate_heatmaps.py`** (289 lines) - Main script
   - Command-line driven, no YAML needed
   - Auto-discovers H5 files in directory
   - Optimized for 2D H5 format
   - Direct model loading via `src.builder.create_model`

2. **`view_heatmap_sync.py`** (9.3 KB) - Interactive viewer
   - Side-by-side WSI + heatmap synchronized navigation
   - Multi-resolution pyramid support

3. **`HEATMAPS_README.md`** - Clean, concise documentation
   - Quick start examples
   - All command-line options
   - Troubleshooting guide

4. **`example_generate_heatmaps.sh`** - Usage examples with real PANDA paths

### ALTERNATIVE: CLAM-Compatible Implementation

5. **`create_heatmaps.py`** (526 lines) - YAML-based generation
   - Full CLAM compatibility
   - Requires config files
   - More complex but feature-rich

6. **`create_process_list.py`** - Helper to generate slide CSVs

7. **`heatmaps/configs/`** - YAML configuration templates
   - `config_template.yaml` - Generic template
   - `config_abmil_panda.yaml` - PANDA-specific

8. **`HEATMAP_GENERATION.md`**, **`HEATMAP_QUICKSTART.md`** - Comprehensive docs

### Core Utilities (Already Existed)

9. **`src/visualization/heatmap_utils.py`** (19 KB) - Core visualization code
   - `WholeSlideImage` class - WSI wrapper with tissue segmentation
   - `draw_heatmap()` - Create heatmap overlay
   - `sample_top_patches()` - Extract high-attention patches
   - `MILLabVisualizer` - High-level interface
   - `to_percentiles()` - Score normalization

10. **`src/visualization/__init__.py`** - Module exports

11. **`utils/file_utils.py`** - HDF5/pickle I/O utilities
    - `save_hdf5()`, `load_hdf5()`
    - `save_pkl()`, `load_pkl()`

12. **`test_heatmap_setup.py`** - Dependency verification script

## Quick Usage (RECOMMENDED)

```bash
# Generate heatmaps for all slides
python generate_heatmaps.py \
    --checkpoint checkpoints/abmil_panda_best.pt \
    --h5_dir /media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/ \
    --slide_dir /media/nadim/Data/prostate-cancer-grade-assessment/train_images \
    --output_dir heatmaps/panda_results/ \
    --slide_ext .tiff

# Test with first 10 slides
python generate_heatmaps.py ... --limit 10

# Different visualization
python generate_heatmaps.py ... --cmap plasma --alpha 0.5

# View results interactively
python view_heatmap_sync.py \
    --slide /path/to/slide.tiff \
    --heatmap heatmaps/panda_results/slide_001_heatmap.png
```

## Output Structure

```
heatmaps/panda_results/
├── slide_001_heatmap.png       # Heatmap visualization
├── slide_002_heatmap.png
├── ...
└── heatmap_results.csv         # Results summary
```

**CSV contains:**
- slide_id, num_patches, prediction, probabilities
- attention_mean, attention_std, heatmap_path

## Key Implementation Details

### Model Loading
Uses MIL-Lab's `create_model()` function:
```python
from src.builder import create_model

model = create_model(
    model_name='abmil.base.uni_v2.pc108-24k',  # Standard naming convention
    num_classes=2,
    checkpoint_path='path/to/checkpoint.pt'
).to(device)
```

### Attention Extraction
Models return attention via `return_attention=True`:
```python
results_dict, log_dict = model(features, return_attention=True)
attention = log_dict['attention'].cpu().numpy().squeeze()
```

### H5 Loading Pattern
Handles both 2D (new) and 3D (old) formats:
```python
with h5py.File(h5_path, 'r') as f:
    raw_features = torch.from_numpy(f['features'][:])
    features = raw_features.squeeze(0).clone()  # Works for both 2D and 3D
    coords = f['coords'][:]
```

### Tissue Segmentation
Automatically segments tissue regions to mask heatmap:
```python
wsi.segment_tissue(
    seg_level=-1,  # Auto-select level
    sthresh=15,    # Saturation threshold
    mthresh=11,    # Median blur
    close=2,       # Morphological closing
    filter_params={'a_t': 50, 'a_h': 8, 'max_n_holes': 10}
)
```

### Heatmap Generation
Core function from `src.visualization`:
```python
heatmap = draw_heatmap(
    scores=attention,              # (N,) attention scores
    coords=coords,                 # (N, 2) coordinates at level 0
    slide_path=slide_path,
    wsi_object=wsi,
    vis_level=-1,                  # Auto-select visualization level
    patch_size=(256, 256),
    cmap='jet',                    # jet, coolwarm, viridis, plasma
    alpha=0.4,                     # Transparency 0-1
    convert_to_percentiles=True,   # Better contrast
    segment=True                   # Use tissue mask
)
```

## Project Integration

### Existing Workflow
```
Feature Extraction → Training → Evaluation
     (Trident)       (MIL-Lab)   (metrics)
```

### New Workflow
```
Feature Extraction → Training → Evaluation → Heatmap Generation
     (Trident)       (MIL-Lab)   (metrics)      (NEW!)
```

### Data Flow
```
WSI (.tiff)
  → Trident → slide_id.h5 (features + coords)
  → Training → checkpoint.pt (trained model)
  → Heatmap → slide_id_heatmap.png + results.csv
```

## Dependencies (All Installed)

- torch, h5py, numpy, pandas
- openslide-python (WSI reading)
- matplotlib (colormaps)
- scipy (percentile calculations)
- PIL, cv2 (image processing)
- tqdm (progress bars)

**Test setup:** `python test_heatmap_setup.py` (all tests passing ✓)

## Common Parameters

### Colormaps
- `jet` - Blue→Red (classic, high contrast) - DEFAULT
- `coolwarm` - Blue→White→Red (diverging)
- `viridis` - Purple→Green→Yellow (perceptually uniform)
- `plasma` - Purple→Pink→Yellow (vibrant)

### Visualization Levels
- `-1` - Auto (typically ~32x downsample) - DEFAULT
- `0` - Full resolution (slow, large files)
- `1-4` - Progressively lower resolution (faster)

### Transparency (alpha)
- `0.0` - Show only original image
- `0.4` - Balanced blend - DEFAULT
- `1.0` - Show only heatmap

## Troubleshooting

**Model loading fails:**
- Check model_name matches builder conventions: `{type}.{size}.{encoder}.{postfix}`
- Example: `abmil.base.uni_v2.pc108-24k`

**H5 files not found:**
- Verify h5_dir path is correct
- Ensure files are named `{slide_id}.h5`

**Slide files not found:**
- Verify slide_dir path is correct
- Check slide_ext matches (`.tiff`, `.svs`, etc.)

**Out of memory:**
- Use lower resolution: `--vis_level 3`
- Process fewer slides: `--limit 100`

## Code Philosophy

**Simple version (generate_heatmaps.py):**
- ✅ Lean: 289 lines
- ✅ Readable: Clear function names, inline comments
- ✅ Direct: Command-line args, no configs
- ✅ Modern: Optimized for current H5 format
- ✅ Minimal: Only essential features

**CLAM version (create_heatmaps.py):**
- Full-featured: 526 lines
- YAML-based configuration
- Backward compatible with CLAM
- More abstraction layers
- Use when you need CLAM compatibility

## Testing

```bash
# 1. Test dependencies
python test_heatmap_setup.py

# 2. Test with small dataset
python generate_heatmaps.py \
    --checkpoint checkpoints/test.pt \
    --h5_dir /path/to/h5/ \
    --slide_dir /path/to/slides/ \
    --output_dir heatmaps/test/ \
    --limit 5  # Just 5 slides

# 3. Check outputs
ls heatmaps/test/
cat heatmaps/test/heatmap_results.csv
```

## Next Steps / Future Enhancements

Potential additions (not implemented):
- [ ] Batch processing with multiprocessing
- [ ] ROI-specific heatmaps
- [ ] Video generation (sliding window through WSI)
- [ ] Patch extraction based on attention threshold
- [ ] Heatmap comparison between models
- [ ] Integration with MLflow/Weights&Biases logging

## Important Notes

1. **Prefer simple version** for day-to-day use
2. **H5 format is 2D** - code handles this directly
3. **Coordinates are at level 0** - heatmap code handles scaling
4. **Model must support `return_attention=True`** - all MIL-Lab models do
5. **Tissue segmentation is automatic** - can disable with `segment=False`

## File Locations Summary

```
MIL-Lab/
├── generate_heatmaps.py           # MAIN SCRIPT (simple)
├── view_heatmap_sync.py           # Interactive viewer
├── HEATMAPS_README.md             # MAIN DOCS (simple)
├── example_generate_heatmaps.sh   # Examples
├── test_heatmap_setup.py          # Dependency test
│
├── create_heatmaps.py             # Alternative (CLAM-style)
├── create_process_list.py         # Helper (CLAM-style)
├── HEATMAP_GENERATION.md          # Comprehensive docs (CLAM)
├── HEATMAP_QUICKSTART.md          # Quick guide (CLAM)
│
├── src/visualization/
│   ├── heatmap_utils.py           # Core visualization utilities
│   └── __init__.py
│
├── utils/
│   └── file_utils.py              # HDF5 I/O utilities
│
└── heatmaps/
    ├── configs/
    │   ├── config_template.yaml
    │   └── config_abmil_panda.yaml
    └── process_lists/
        └── example_process_list.csv
```

## Quick Reference Card

```bash
# Generate heatmaps (minimal)
python generate_heatmaps.py \
    --checkpoint MODEL.pt \
    --h5_dir H5_DIR \
    --slide_dir SLIDE_DIR \
    --output_dir OUTPUT

# Common options
--limit 10              # Test with 10 slides
--cmap plasma           # Different colormap
--alpha 0.5             # More/less transparency
--csv slides.csv        # Process specific slides
--slide_ext .svs        # Different slide format
--vis_level 2           # Lower resolution

# View results
python view_heatmap_sync.py --slide SLIDE.tiff --heatmap HEATMAP.png

# Test setup
python test_heatmap_setup.py
```

---

**Status:** ✅ Complete, tested, documented, ready to use
**Recommended:** Use `generate_heatmaps.py` (simple version)
**All tests passing:** Dependencies verified via `test_heatmap_setup.py`
