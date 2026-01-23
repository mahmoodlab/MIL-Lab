# Visualization Files Cleanup Plan

This document outlines the plan for cleaning up visualization-related files from the top-level directory after migrating to the new `Visualization/` directory structure.

---

## Current State Analysis

### Python Scripts (10 files)

| File | Size | Purpose | Action |
|------|------|---------|--------|
| `generate_heatmaps_highres.py` | 18K | High-res heatmaps with overlapping patches (CLI) | **ARCHIVE** - Replaced by Visualization/generate_heatmaps_highres.py |
| `generate_heatmaps.py` | 11K | Standard heatmaps from pre-computed H5 features | **KEEP** - Different use case (uses pre-computed features) |
| `create_heatmaps.py` | 18K | YAML-based heatmap generation (CLAM-inspired) | **EVALUATE** - May move to Visualization/ with YAML support |
| `view_heatmap_sync.py` | 9.3K | Interactive synchronized viewer (WSI + heatmap) | **MOVE** - Useful utility, belongs in Visualization/ |
| `visualize_model.py` | 4.2K | Visualize top patches and heatmaps for models | **MOVE** - General visualization utility |
| `create_process_list.py` | 6.4K | Create batch processing lists | **MOVE** - Helper for batch heatmap generation |
| `debug_heatmap.py` | 6.3K | Debug heatmap generation issues | **MOVE** - Debugging utility |
| `test_heatmap_setup.py` | 5.5K | Test heatmap setup | **MOVE** - Testing utility |
| `test_simple_heatmap.py` | 5.5K | Simple heatmap test | **MOVE** - Testing utility |

### Shell Scripts (2 files)

| File | Size | Purpose | Action |
|------|------|---------|--------|
| `example_generate_heatmaps.sh` | 1.7K | Example batch heatmap generation | **MOVE** - Example belongs with Visualization/ |
| `example_highres_heatmap.sh` | 1.8K | Example high-res heatmap | **MOVE** - Example belongs with Visualization/ |

### Documentation (7 files)

| File | Size | Purpose | Action |
|------|------|---------|--------|
| `HEATMAPS_README.md` | 4.9K | Main heatmap guide | **CONSOLIDATE** - Merge into Visualization/README.md |
| `HEATMAP_QUICKSTART.md` | 7.8K | Quick start guide | **CONSOLIDATE** - Already have Visualization/QUICK_START.md |
| `HEATMAP_GENERATION.md` | 12K | Detailed generation process | **CONSOLIDATE** - Merge relevant parts to Visualization/README.md |
| `HEATMAP_FILES_SUMMARY.md` | 9.9K | Summary of heatmap files | **ARCHIVE** - Outdated after reorganization |
| `HEATMAP_SUMMARY.md` | 12K | High-level overview | **CONSOLIDATE** - Merge into Visualization/README.md |
| `HEATMAP_CHECKLIST.md` | 2.3K | Checklist for generation | **MOVE** - Useful checklist |
| `HIGHRES_HEATMAPS.md` | 9.0K | High-res heatmap guide | **CONSOLIDATE** - Already covered in Visualization/README.md |

---

## Proposed Directory Structure

```
MIL-Lab/
├── Visualization/
│   ├── generate_heatmaps_highres.py  # NEW: JSON-based high-res heatmaps
│   ├── generate_heatmaps.py          # MOVED: Pre-computed features version
│   ├── create_heatmaps.py            # MOVED: YAML-based version
│   ├── view_heatmap_sync.py          # MOVED: Interactive viewer
│   ├── visualize_model.py            # MOVED: Model visualization
│   ├── create_process_list.py        # MOVED: Batch processing helper
│   │
│   ├── debug/                        # NEW: Debug scripts subdirectory
│   │   ├── debug_heatmap.py
│   │   ├── test_heatmap_setup.py
│   │   └── test_simple_heatmap.py
│   │
│   ├── examples/                     # NEW: Example scripts
│   │   ├── example_generate_heatmaps.sh
│   │   └── example_highres_heatmap.sh
│   │
│   ├── config_example.json           # EXISTS
│   ├── config_high_overlap.json      # EXISTS
│   ├── config_fast.json              # EXISTS
│   │
│   ├── README.md                     # EXISTS: Comprehensive guide
│   ├── QUICK_START.md                # EXISTS: Quick reference
│   └── HEATMAP_CHECKLIST.md          # MOVED: Useful checklist
│
└── archive/                          # NEW: Archived old files
    └── visualization_legacy/
        ├── generate_heatmaps_highres.py  # Old CLI version
        ├── HEATMAPS_README.md            # Old docs
        ├── HEATMAP_QUICKSTART.md
        ├── HEATMAP_GENERATION.md
        ├── HEATMAP_FILES_SUMMARY.md
        ├── HEATMAP_SUMMARY.md
        └── HIGHRES_HEATMAPS.md
```

---

## Action Categories

### 1. MOVE to Visualization/ (9 files)

**Scripts to move as-is:**
- `generate_heatmaps.py` - Different use case (pre-computed features)
- `create_heatmaps.py` - YAML-based version
- `view_heatmap_sync.py` - Interactive viewer
- `visualize_model.py` - Model visualization
- `create_process_list.py` - Batch helper

**Move to Visualization/debug/:**
- `debug_heatmap.py`
- `test_heatmap_setup.py`
- `test_simple_heatmap.py`

**Move to Visualization/examples/:**
- `example_generate_heatmaps.sh`
- `example_highres_heatmap.sh`

**Move to Visualization/:**
- `HEATMAP_CHECKLIST.md` - Still useful

### 2. ARCHIVE (8 files)

**Move to archive/visualization_legacy/:**
- `generate_heatmaps_highres.py` - Replaced by JSON version
- `HEATMAPS_README.md` - Consolidated into new docs
- `HEATMAP_QUICKSTART.md` - Replaced by QUICK_START.md
- `HEATMAP_GENERATION.md` - Consolidated
- `HEATMAP_FILES_SUMMARY.md` - Outdated
- `HEATMAP_SUMMARY.md` - Consolidated
- `HIGHRES_HEATMAPS.md` - Consolidated

### 3. KEEP in Root (0 files)

All visualization files should be in Visualization/ directory.

---

## Step-by-Step Cleanup Process

### Phase 1: Create Subdirectories
```bash
mkdir -p Visualization/debug
mkdir -p Visualization/examples
mkdir -p archive/visualization_legacy
```

### Phase 2: Move Scripts
```bash
# Move main scripts
mv generate_heatmaps.py Visualization/
mv create_heatmaps.py Visualization/
mv view_heatmap_sync.py Visualization/
mv visualize_model.py Visualization/
mv create_process_list.py Visualization/

# Move debug scripts
mv debug_heatmap.py Visualization/debug/
mv test_heatmap_setup.py Visualization/debug/
mv test_simple_heatmap.py Visualization/debug/

# Move examples
mv example_generate_heatmaps.sh Visualization/examples/
mv example_highres_heatmap.sh Visualization/examples/

# Move checklist
mv HEATMAP_CHECKLIST.md Visualization/
```

### Phase 3: Archive Old Files
```bash
# Archive replaced script
mv generate_heatmaps_highres.py archive/visualization_legacy/

# Archive old documentation
mv HEATMAPS_README.md archive/visualization_legacy/
mv HEATMAP_QUICKSTART.md archive/visualization_legacy/
mv HEATMAP_GENERATION.md archive/visualization_legacy/
mv HEATMAP_FILES_SUMMARY.md archive/visualization_legacy/
mv HEATMAP_SUMMARY.md archive/visualization_legacy/
mv HIGHRES_HEATMAPS.md archive/visualization_legacy/
```

### Phase 4: Update Documentation
```bash
# Update Visualization/README.md with any missing info from old docs
# Update SCRIPTS_SUMMARY.md to reflect new structure
# Create Visualization/debug/README.md for debug scripts
# Create archive/visualization_legacy/README.md explaining archived files
```

### Phase 5: Fix Import Paths
```bash
# Update imports in moved scripts to work from new location
# Most scripts should work with: sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Phase 6: Update References
```bash
# Search for references to moved files in other scripts
# Update any hardcoded paths in example scripts
```

---

## Benefits of Cleanup

### Organization
- **Before**: 19 visualization files scattered in root directory
- **After**: All visualization in `Visualization/` with logical subdirectories

### Discoverability
- Clear separation: scripts vs examples vs debug vs docs
- New users know exactly where to look

### Maintenance
- Easier to maintain related files together
- Version control is cleaner (logical grouping)

### Backward Compatibility
- Archived files preserved in `archive/` if needed
- Can create symlinks if necessary for legacy scripts

---

## Testing Checklist

After cleanup, verify:

- [ ] Visualization/generate_heatmaps_highres.py works with JSON config
- [ ] Visualization/generate_heatmaps.py works from new location
- [ ] Visualization/create_heatmaps.py works from new location
- [ ] Visualization/view_heatmap_sync.py works from new location
- [ ] Example scripts in Visualization/examples/ run correctly
- [ ] Debug scripts in Visualization/debug/ work
- [ ] All documentation is accessible and accurate
- [ ] SCRIPTS_SUMMARY.md reflects new structure
- [ ] No broken imports in other scripts

---

## Rollback Plan

If issues arise, rollback is simple:

```bash
# Restore from archive
cp -r archive/visualization_legacy/* .

# Remove Visualization contents
rm -rf Visualization/debug Visualization/examples
# Keep Visualization/generate_heatmaps_highres.py and configs
```

---

## Timeline

Estimated time: **15-30 minutes**

1. Create directories: 1 min
2. Move files: 5 min
3. Update imports: 5-10 min
4. Update documentation: 5-10 min
5. Test: 5-10 min

---

## Questions to Consider

1. **Should we keep YAML support in create_heatmaps.py?**
   - Pros: YAML is human-readable, CLAM-compatible
   - Cons: Now have both JSON and YAML configs
   - **Recommendation**: Keep for now, mark as legacy in docs

2. **Should debug scripts go in a separate debug/ subdirectory?**
   - Pros: Cleaner organization
   - Cons: One more level of nesting
   - **Recommendation**: Yes, helps separate debugging from production tools

3. **Should we create a utilities/ subdirectory?**
   - For scripts like `create_process_list.py`, `view_heatmap_sync.py`
   - **Recommendation**: Not necessary yet, main directory is fine

4. **Archive location: archive/ vs docs/archive/?**
   - **Recommendation**: Top-level `archive/` for visibility

---

## Post-Cleanup Documentation Updates

### Files to update:

1. **SCRIPTS_SUMMARY.md**
   - Remove top-level visualization entries
   - Expand Visualization Directory section
   - Add subdirectory descriptions

2. **Visualization/README.md**
   - Add sections on debug scripts
   - Add sections on example scripts
   - Consolidate info from archived docs

3. **Create: Visualization/debug/README.md**
   - Explain debug scripts
   - When to use each one

4. **Create: Visualization/examples/README.md**
   - Explain example scripts
   - How to customize them

5. **Create: archive/visualization_legacy/README.md**
   - Explain what's archived
   - Why files were replaced
   - Migration guide if needed

---

**Ready to execute upon user approval after testing the new script.**
