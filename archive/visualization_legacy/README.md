# Archived Visualization Files

This directory contains legacy visualization files that have been replaced by the new `Visualization/` directory structure.

## Why These Files Are Archived

The visualization codebase has been reorganized and improved. These files are preserved for reference and potential recovery if needed.

## What's Archived

### Scripts

**generate_heatmaps_highres.py** - Legacy CLI-based high-resolution heatmap generator
- **Replaced by:** `Visualization/generate_heatmaps_highres.py` (JSON config version)
- **Key improvements in new version:**
  - JSON configuration instead of CLI arguments
  - Auto-generated filenames with all parameters
  - Better documentation and example configs
  - Cleaner code organization

### Documentation

**HEATMAPS_README.md** - Old main heatmap documentation
- **Replaced by:** `Visualization/README.md`
- **Improvements:** Consolidated information, better organization, JSON config focus

**HEATMAP_QUICKSTART.md** - Old quick start guide
- **Replaced by:** `Visualization/QUICK_START.md`
- **Improvements:** Updated for JSON configs, clearer examples, better formatting

**HEATMAP_GENERATION.md** - Detailed generation process documentation
- **Status:** Key information merged into `Visualization/README.md`

**HEATMAP_FILES_SUMMARY.md** - Summary of heatmap-related files
- **Status:** Outdated after reorganization

**HEATMAP_SUMMARY.md** - High-level overview
- **Status:** Consolidated into new documentation

**HIGHRES_HEATMAPS.md** - High-resolution heatmap guide
- **Status:** Fully covered in `Visualization/README.md` and `QUICK_START.md`

## Migration Guide

### If you were using `generate_heatmaps_highres.py` (old CLI version):

**Old command:**
```bash
python generate_heatmaps_highres.py \
    --checkpoint model.pt \
    --slide slide.tiff \
    --output heatmap.tiff \
    --overlap 0.90 \
    --cmap jet
```

**New approach:**
1. Create a JSON config file:
```json
{
  "checkpoint": "model.pt",
  "slide": "slide.tiff",
  "output": "output_dir/",
  "overlap": 0.90,
  "cmap": "jet"
}
```

2. Run with config:
```bash
cd Visualization/
python generate_heatmaps_highres.py --config my_config.json
```

**Benefits:**
- Filename auto-generated with all parameters
- Easier to track and reproduce experiments
- Reusable configs
- Better parameter organization

### If you need the old script:

The old script is preserved here if you need it for any reason:
```bash
python ../archive/visualization_legacy/generate_heatmaps_highres.py [old arguments]
```

## Rollback Instructions

If you need to restore the old structure:

```bash
# From MIL-Lab root directory
cp archive/visualization_legacy/*.py .
cp archive/visualization_legacy/*.md .
```

**Note:** Not recommended unless there's a critical issue with the new system.

## Questions?

If you have questions about the migration or need help with the new system:
1. See `Visualization/README.md` for comprehensive documentation
2. See `Visualization/QUICK_START.md` for quick examples
3. Check example configs in `Visualization/config_*.json`

## Timeline

- **Archived:** December 2025
- **Reason:** Codebase reorganization and improvement
- **Status:** Preserved for reference, not actively maintained
