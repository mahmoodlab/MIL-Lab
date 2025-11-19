## High-Resolution Heatmap Generation

Generate ultra high-resolution heatmaps at 20x magnification (or any resolution) with overlapping patch inference.

### Quick Start

```bash
# Feature extractor is auto-detected from checkpoint!
python generate_heatmaps_highres.py \
    --checkpoint best_model_abmil.pth \
    --slide /path/to/slide.tiff \
    --output heatmap_highres.tiff \
    --overlap 0.90
```

### How It Works

**Standard Heatmap** (from `generate_heatmaps.py`):
- Uses pre-extracted features from Trident (no overlap)
- Fast but lower resolution
- Example: 10,000 patches for a slide

**High-Resolution Heatmap** (from `generate_heatmaps_highres.py`):
- Creates dense grid of overlapping patches on-the-fly
- Extracts features and gets attention for each patch
- Much higher resolution but slower
- Example: 100,000+ patches for same slide with 90% overlap

### Overlap Settings

| Overlap | Step Size | Patches (relative) | Use Case |
|---------|-----------|-------------------|----------|
| 0% | 256px | 1x (baseline) | Standard resolution |
| 50% | 128px | 4x | Moderate increase |
| 90% | 26px | 100x | High resolution |
| 95% | 13px | 400x | Ultra high resolution |

**Calculation**: `step_size = patch_size * (1 - overlap)`
- Patch size: 256px
- 90% overlap → step_size = 256 * 0.1 = 25.6px ≈ 26px
- 95% overlap → step_size = 256 * 0.05 = 12.8px ≈ 13px

### Parameters

```bash
# Required
--checkpoint PATH        # Model checkpoint
--slide PATH            # WSI file (.tiff, .svs, etc.)
--output PATH           # Output path (.tiff or .png)

# Feature extraction (NEW: Auto-detection!)
--feature_extractor auto   # auto (default), uni_v1, uni_v2, resnet50
                          # 'auto' detects from checkpoint automatically

# Key parameters
--overlap 0.90          # Overlap ratio (default: 0.90 = 90%)
--vis_level 0           # Output resolution level (0=full, 1=2x downsample)
--patch_size 256        # Patch size in pixels

# Visualization
--cmap jet              # Colormap: jet, coolwarm, viridis, plasma
--alpha 0.5             # Transparency 0-1

# Performance tuning
--batch_size_extract 32    # Batch size for feature extraction
--batch_size_infer 512     # Batch size for attention inference
```

### Automatic Feature Extractor Detection

**The script now automatically detects which feature extractor to use!**

By default (`--feature_extractor auto`), the script:
1. Reads your model checkpoint
2. Detects the feature dimension from `patch_embed` layer
3. Automatically loads the matching feature extractor

**Dimension → Extractor Mapping:**
- **1536-dim** → UNI v2 (UNI2-h)
- **1024-dim** → UNI v1 or ResNet50
- **768-dim** → CTransPath (requires Trident)
- **512-dim** → CONCH v1 (requires Trident)
- **2560-dim** → Virchow (requires Trident)

**Manual override:**
```bash
# Force specific extractor (bypasses auto-detection)
--feature_extractor uni_v2    # Use UNI v2 regardless of checkpoint
--feature_extractor uni_v1    # Use UNI v1
--feature_extractor resnet50  # Use ResNet50
```

### Output Files

For each slide, generates:
1. **`output.tiff`** - High-resolution heatmap (TIFF or PNG)
2. **`output_coords.h5`** - Dense coordinates and attention scores

The H5 file contains:
- `coords`: (N, 2) patch coordinates at level 0
- `attention_scores`: (N, 1) attention scores

### Resolution Comparison

**Example: Slide dimensions 50,000 x 50,000 pixels**

| Method | Patches | Resolution | Speed |
|--------|---------|-----------|-------|
| Standard (0% overlap) | ~10,000 | 256px grid | Fast |
| 90% overlap | ~100,000 | 26px grid | 10x slower |
| 95% overlap | ~400,000 | 13px grid | 40x slower |

### Memory and Speed

**GPU Memory:**
- Feature extraction: ~batch_size_extract * 3 * 224 * 224 * 4 bytes
- Attention inference: ~batch_size_infer * feature_dim * 4 bytes
- Default settings work with 12GB VRAM

**Speed estimates** (approximate, depends on hardware):
- Standard (0% overlap): ~30 seconds per slide
- 90% overlap: ~5 minutes per slide
- 95% overlap: ~20 minutes per slide

**If you run out of memory:**
```bash
# Reduce extraction batch size
--batch_size_extract 16

# Or use CPU for extraction (slower but more memory)
# (requires code modification to move extractor to CPU)
```

### Examples

**1. Standard high-res with auto-detection (RECOMMENDED)**
```bash
# No need to specify feature extractor - auto-detected!
python generate_heatmaps_highres.py \
    --checkpoint best_model.pth \
    --slide slide.tiff \
    --output heatmap_highres.tiff \
    --overlap 0.90
```

**2. Ultra high-res (95% overlap)**
```bash
python generate_heatmaps_highres.py \
    --checkpoint best_model.pth \
    --slide slide.tiff \
    --output heatmap_ultrahires.tiff \
    --overlap 0.95 \
    --vis_level 0
```

**3. Balanced (90% overlap, 2x downsampled - faster)**
```bash
python generate_heatmaps_highres.py \
    --checkpoint best_model.pth \
    --slide slide.tiff \
    --output heatmap_balanced.tiff \
    --overlap 0.90 \
    --vis_level 1  # 2x downsample
```

**4. Different colormap**
```bash
python generate_heatmaps_highres.py \
    --checkpoint best_model.pth \
    --slide slide.tiff \
    --output heatmap_plasma.tiff \
    --overlap 0.90 \
    --cmap plasma \
    --alpha 0.6
```

**5. Manual feature extractor override**
```bash
# Force UNI v1 even if checkpoint expects different dimension
python generate_heatmaps_highres.py \
    --checkpoint best_model.pth \
    --slide slide.tiff \
    --output heatmap.tiff \
    --feature_extractor uni_v1  # Override auto-detection
    --overlap 0.90
```

### When to Use Each Method

**Use `generate_heatmaps.py` (standard) when:**
- ✅ You want quick heatmaps for many slides
- ✅ You already have pre-extracted features
- ✅ Standard resolution is sufficient
- ✅ You're exploring/debugging

**Use `generate_heatmaps_highres.py` (high-res) when:**
- ✅ You need publication-quality heatmaps
- ✅ You want to see fine-grained attention patterns
- ✅ You're analyzing specific slides in detail
- ✅ You want 20x resolution output

### Workflow

```bash
# 1. Quick exploration - use standard method
python generate_heatmaps.py \
    --checkpoint model.pth \
    --h5_dir embeddings/ \
    --slide_dir slides/ \
    --output_dir heatmaps/quick/ \
    --limit 100

# 2. Identify interesting slides from results.csv

# 3. Generate high-res for specific slides
python generate_heatmaps_highres.py \
    --checkpoint model.pth \
    --slide slides/interesting_slide.tiff \
    --output heatmaps/highres/interesting_slide.tiff \
    --overlap 0.90

# 4. View in OpenSlide viewer
python view_heatmap_sync.py \
    --slide slides/interesting_slide.tiff \
    --heatmap heatmaps/highres/interesting_slide.tiff
```

### Technical Details

**Feature Extraction:**
- Uses UNI v2 (UNI2-h) feature extractor by default
  - **Important**: Loads `hf-hub:MahmoodLab/UNI2-h` to get 1536-dimensional features
  - This matches the features extracted by Trident for training
  - The standard `MahmoodLab/uni` model produces 1024-dim features (different model!)
- Transforms: Resize to 224x224, center crop, normalize
- Extracted on-the-fly for each patch in dense grid

**Attention Computation:**
- Processes features in batches through MIL model
- Gets attention scores using `return_attention=True`
- Applies softmax normalization

**Heatmap Creation:**
- Uses same `draw_heatmap()` function as standard method
- Overlapping patches create smooth, high-resolution output
- Percentile normalization for better contrast

### Tips

1. **Start with 90% overlap** - Good balance of quality and speed
2. **Use vis_level=1 for testing** - 2x faster, still good quality
3. **Save coords.h5** - Enables later analysis without re-processing
4. **TIFF format for high-res** - Better compression than PNG for large images
5. **Batch processing** - Process one slide at a time to avoid memory issues

### Troubleshooting

**"Dimension mismatch" or "mat1 and mat2 shapes cannot be multiplied"**
- **Root cause**: Model was trained with 1536-dim UNI v2 features, but script is using wrong feature extractor
- **Solution**: The script now correctly loads `UNI2-h` (1536-dim) instead of standard `uni` (1024-dim)
- This was fixed in generate_heatmaps_highres.py at lines 89-124
- If you see this error, make sure you have the latest version of the script

**"Out of memory during feature extraction"**
```bash
--batch_size_extract 16  # or even 8
```

**"Out of memory during attention inference"**
```bash
--batch_size_infer 256  # or 128
```

**"Takes too long"**
- Use lower overlap: `--overlap 0.85` or `--overlap 0.80`
- Use higher vis_level: `--vis_level 1` or `--vis_level 2`
- Consider whether you really need high-res for all slides

**"Heatmap looks pixelated"**
- Increase overlap: `--overlap 0.95`
- Use lower vis_level: `--vis_level 0`
- Enable blur in the heatmap drawing code

---

**Summary:**
- **Standard method**: Fast, uses pre-extracted features, good for batch processing
- **High-res method**: Slow, extracts features on-the-fly, publication quality

Both methods use the same visualization code and produce compatible outputs!
