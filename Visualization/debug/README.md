# Debug Scripts

This directory contains debugging and testing utilities for heatmap generation.

## Scripts

### debug_heatmap.py

Debug script to diagnose heatmap generation issues.

**Usage:**
```bash
python debug_heatmap.py \
    --checkpoint path/to/model.pt \
    --h5_file path/to/slide.h5 \
    --slide path/to/slide.tiff
```

**Purpose:**
- Check model loading
- Verify H5 file format
- Test attention score generation
- Debug coordinate alignment issues

### test_heatmap_setup.py

Test script to verify all dependencies and imports are correctly configured.

**Usage:**
```bash
python test_heatmap_setup.py
```

**Tests:**
- PyTorch installation
- OpenSlide availability
- H5py functionality
- PIL/Pillow installation
- Custom MIL-Lab modules (src.visualization, src.builder)

### test_simple_heatmap.py

Simple test to verify basic heatmap drawing functionality.

**Usage:**
```bash
python test_simple_heatmap.py \
    --h5_file path/to/slide.h5 \
    --slide path/to/slide.tiff \
    --output test_heatmap.png
```

**Purpose:**
- Test basic heatmap generation without model
- Verify coordinate loading from H5
- Test heatmap overlay rendering
- Quick sanity check for visualization pipeline

## When to Use

- **Before generating heatmaps**: Run `test_heatmap_setup.py` to verify environment
- **If heatmaps fail**: Use `debug_heatmap.py` to diagnose the issue
- **To test visualization**: Use `test_simple_heatmap.py` for quick rendering tests

## Common Issues

1. **Import errors**: Make sure you're running from the Visualization directory
2. **OpenSlide not found**: Install openslide library for your OS
3. **H5 format mismatch**: Check that embeddings match expected format (N, D)
4. **Coordinate alignment**: Verify patch coordinates match embedding order
