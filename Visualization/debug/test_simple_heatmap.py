#!/usr/bin/env python3
"""
Simple test to verify heatmap drawing works

Usage:
    python test_simple_heatmap.py \
        --h5_file path/to/slide.h5 \
        --slide path/to/slide.tiff \
        --output test_heatmap.png
"""

import sys
from pathlib import Path

# Add grandparent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src.visualization import WholeSlideImage, draw_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_file', type=str, required=True)
    parser.add_argument('--slide', type=str, required=True)
    parser.add_argument('--output', type=str, default='test_heatmap.png')
    args = parser.parse_args()

    print("="*70)
    print("SIMPLE HEATMAP TEST")
    print("="*70)

    # Load coordinates
    print("\n1. Loading coordinates...")
    with h5py.File(args.h5_file, 'r') as f:
        coords = f['coords'][:]
        num_patches = len(coords)

    print(f"   Found {num_patches} patches")
    print(f"   Coord range: X=[{coords[:, 0].min()}, {coords[:, 0].max()}]")
    print(f"                Y=[{coords[:, 1].min()}, {coords[:, 1].max()}]")

    # Create FAKE attention scores - random for testing
    print("\n2. Creating fake attention scores (random)...")
    np.random.seed(42)
    fake_attention = np.random.rand(num_patches)
    fake_attention = fake_attention / fake_attention.sum()  # Normalize

    print(f"   Attention range: [{fake_attention.min():.6f}, {fake_attention.max():.6f}]")
    print(f"   Attention sum: {fake_attention.sum():.6f}")
    print(f"   Top 5 values: {np.sort(fake_attention)[-5:]}")

    # Initialize WSI
    print("\n3. Loading slide...")
    wsi = WholeSlideImage(args.slide)
    print(f"   Slide: {wsi.name}")
    print(f"   Dimensions: {wsi.level_dim[0]}")
    print(f"   Levels: {wsi.level_count}")

    # Segment tissue
    print("\n4. Segmenting tissue...")
    wsi.segment_tissue(
        seg_level=-1,
        sthresh=15,
        mthresh=11,
        close=2,
        filter_params={'a_t': 50, 'a_h': 8, 'max_n_holes': 10}
    )
    print(f"   Found {len(wsi.contours_tissue)} tissue regions")

    # Create heatmap
    print("\n5. Drawing heatmap...")
    heatmap = draw_heatmap(
        scores=fake_attention,
        coords=coords,
        slide_path=args.slide,
        wsi_object=wsi,
        vis_level=-1,
        patch_size=(256, 256),
        cmap='jet',
        alpha=0.6,  # Higher alpha for more visible heatmap
        convert_to_percentiles=True,
        segment=True,
        blank_canvas=False
    )

    # Save
    print(f"\n6. Saving to {args.output}...")
    heatmap.save(args.output)
    print(f"   ✓ Saved!")
    print(f"   Image size: {heatmap.size}")

    # Also save a blank canvas version for comparison
    print("\n7. Creating blank canvas version for comparison...")
    heatmap_blank = draw_heatmap(
        scores=fake_attention,
        coords=coords,
        slide_path=args.slide,
        wsi_object=wsi,
        vis_level=-1,
        patch_size=(256, 256),
        cmap='jet',
        alpha=1.0,  # Full opacity
        convert_to_percentiles=True,
        segment=False,  # No tissue masking
        blank_canvas=True  # Blank canvas to see pure heatmap
    )

    blank_output = args.output.replace('.png', '_blank.png')
    heatmap_blank.save(blank_output)
    print(f"   ✓ Saved blank canvas version to {blank_output}")

    # Create visualization of attention distribution
    print("\n8. Creating attention distribution plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    axes[0, 0].hist(fake_attention, bins=50, edgecolor='black')
    axes[0, 0].set_title('Attention Score Distribution')
    axes[0, 0].set_xlabel('Attention Score')
    axes[0, 0].set_ylabel('Count')

    # Spatial scatter
    axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=fake_attention, s=1, cmap='jet')
    axes[0, 1].set_title('Spatial Attention Distribution')
    axes[0, 1].set_xlabel('X coordinate')
    axes[0, 1].set_ylabel('Y coordinate')
    cb = plt.colorbar(axes[0, 1].scatter(coords[:, 0], coords[:, 1], c=fake_attention, s=1, cmap='jet'), ax=axes[0, 1])
    cb.set_label('Attention')

    # Sorted values
    sorted_attn = np.sort(fake_attention)
    axes[1, 0].plot(sorted_attn)
    axes[1, 0].set_title('Sorted Attention Scores')
    axes[1, 0].set_xlabel('Patch Index (sorted)')
    axes[1, 0].set_ylabel('Attention Score')
    axes[1, 0].grid(True, alpha=0.3)

    # Top patches
    top_k = 10
    top_indices = np.argsort(fake_attention)[-top_k:]
    axes[1, 1].bar(range(top_k), fake_attention[top_indices])
    axes[1, 1].set_title(f'Top {top_k} Attention Scores')
    axes[1, 1].set_xlabel('Rank')
    axes[1, 1].set_ylabel('Attention Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    stats_output = args.output.replace('.png', '_stats.png')
    plt.savefig(stats_output, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved attention stats to {stats_output}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(f"  1. {args.output} - Heatmap on original slide")
    print(f"  2. {blank_output} - Heatmap on blank canvas (pure heatmap)")
    print(f"  3. {stats_output} - Attention statistics")
    print("\nIf you see NO heatmap colors:")
    print("  - Check if patches are within slide bounds")
    print("  - Check if tissue segmentation is working")
    print("  - Try opening the blank canvas version")


if __name__ == '__main__':
    main()
