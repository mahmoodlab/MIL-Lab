# Example Scripts

This directory contains example shell scripts demonstrating heatmap generation workflows.

## Scripts

### example_highres_heatmap.sh

Example of high-resolution heatmap generation with overlapping patches using JSON configuration.

**Features demonstrated:**
- JSON-based configuration
- High overlap (90%) for smooth heatmaps
- Auto-generated filenames from config parameters
- Feature extractor auto-detection

**Usage:**
```bash
# Edit paths in the script first
bash example_highres_heatmap.sh
```

**What it does:**
1. Creates a JSON config file with your paths
2. Runs `generate_heatmaps_highres.py` with the config
3. Produces a high-resolution TIFF heatmap with descriptive filename

**Customize:**
- Edit checkpoint path
- Edit slide path
- Edit output directory
- Adjust overlap (0.70-0.95)
- Change colormap
- Adjust visualization level

### example_generate_heatmaps.sh

Example of standard heatmap generation from pre-computed H5 embeddings.

**Features demonstrated:**
- Batch processing multiple slides
- Using pre-computed feature embeddings
- CSV-based slide lists
- Standard heatmap workflow

**Usage:**
```bash
# Edit paths in the script first
bash example_generate_heatmaps.sh
```

**What it does:**
1. Uses existing H5 embeddings (no on-the-fly feature extraction)
2. Generates attention scores from MIL model
3. Creates heatmap overlays for multiple slides
4. Saves outputs to specified directory

**Customize:**
- Edit checkpoint path
- Edit H5 directory path
- Edit slide directory path
- Edit output directory
- Adjust model parameters

## How to Use These Examples

1. **Copy the example:**
   ```bash
   cp example_highres_heatmap.sh my_heatmap_run.sh
   ```

2. **Edit the paths:**
   ```bash
   nano my_heatmap_run.sh
   # Update checkpoint, slide, and output paths
   ```

3. **Make executable:**
   ```bash
   chmod +x my_heatmap_run.sh
   ```

4. **Run:**
   ```bash
   bash my_heatmap_run.sh
   ```

## Choosing the Right Example

| Use Case | Script | Method |
|----------|--------|--------|
| Single slide, ultra-high quality | `example_highres_heatmap.sh` | On-the-fly extraction + high overlap |
| Batch processing, have embeddings | `example_generate_heatmaps.sh` | Pre-computed H5 embeddings |
| Custom workflow | Create JSON config | `generate_heatmaps_highres.py --config` |

## Tips

- **High-res heatmaps**: Use 90-95% overlap for publication quality
- **Fast previews**: Use 70% overlap and vis_level=1
- **Batch processing**: Create multiple JSON configs and loop through them
- **Memory issues**: Reduce batch sizes in config
