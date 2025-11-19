#!/usr/bin/env python3
"""
Simple Heatmap Generation for MIL-Lab

Generates attention heatmaps from trained MIL models.
Optimized for the new H5 embedding format: (num_patches, feature_dim)

Usage:
    python generate_heatmaps.py \
        --checkpoint path/to/model.pt \
        --h5_dir path/to/embeddings/ \
        --slide_dir path/to/slides/ \
        --output_dir heatmaps/output/
"""

import argparse
import os
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Visualization
from src.visualization import WholeSlideImage, draw_heatmap
from src.builder import create_model


def load_features_and_coords(h5_path):
    """
    Load features and coordinates from H5 file

    Returns:
        features: (N, D) tensor
        coords: (N, 2) array
    """
    with h5py.File(h5_path, 'r') as f:
        # Load features - handle both 2D and 3D formats
        raw_features = torch.from_numpy(f['features'][:])
        features = raw_features.squeeze(0).clone()  # 2D: (N,D) or 3D: (1,N,D) -> (N,D)

        # Load coordinates
        coords = f['coords'][:]

    return features, coords


def get_attention_scores(model, features, device='cuda'):
    """
    Get attention scores from model

    Returns:
        attention: (N,) array of attention scores
        prediction: predicted class
        probabilities: class probabilities
    """
    model.eval()
    features = features.unsqueeze(0).to(device)  # Add batch dimension

    with torch.inference_mode():
        # Run model
        results_dict, log_dict = model(features, return_attention=True)

        # Extract attention
        if 'attention' not in log_dict or log_dict['attention'] is None:
            raise ValueError("Model did not return attention scores. Make sure return_attention=True is working.")

        attention_raw = log_dict['attention'].cpu()

        # Apply softmax to get normalized attention scores (if not already done)
        # ABMIL returns raw scores before softmax
        attention_softmax = torch.softmax(attention_raw, dim=-1)
        attention = attention_softmax.numpy().squeeze()

        # Extract logits and compute predictions
        logits = results_dict.get('logits')
        if logits is not None:
            logits = logits.cpu()
            Y_prob = torch.softmax(logits, dim=-1).numpy().flatten()
            Y_hat = torch.argmax(logits, dim=-1).item()
        else:
            Y_prob = None
            Y_hat = None

    return attention, Y_hat, Y_prob


def generate_heatmap(
    slide_path,
    h5_path,
    model,
    output_path,
    device='cuda',
    cmap='jet',
    alpha=0.4,
    vis_level=-1,
    patch_size=256
):
    """
    Generate heatmap for a single slide

    Args:
        slide_path: Path to WSI file
        h5_path: Path to H5 embeddings file
        model: Trained model
        output_path: Where to save heatmap
        device: cuda or cpu
        cmap: Colormap (jet, coolwarm, viridis, etc.)
        alpha: Transparency (0-1)
        vis_level: Visualization level (-1=auto)
        patch_size: Patch size at level 0
    """
    print(f"  Loading features...")
    features, coords = load_features_and_coords(h5_path)
    print(f"    Features shape: {features.shape}")
    print(f"    Coords shape: {coords.shape}")

    print(f"  Running inference ({len(features)} patches)...")
    attention, pred, probs = get_attention_scores(model, features, device)

    # Match lengths
    min_len = min(len(attention), len(coords))
    attention = attention[:min_len]
    coords = coords[:min_len]

    print(f"  Attention stats:")
    print(f"    Shape: {attention.shape}")
    print(f"    Range: [{attention.min():.6f}, {attention.max():.6f}]")
    print(f"    Mean: {attention.mean():.6f}, Std: {attention.std():.6f}")
    print(f"    Non-zero patches: {(attention > 1e-6).sum()} / {len(attention)}")
    print(f"  Prediction: class {pred}")
    print(f"  Probabilities: {probs}")

    print(f"  Generating heatmap...")
    # Initialize WSI
    wsi = WholeSlideImage(slide_path)

    # Segment tissue
    wsi.segment_tissue(
        seg_level=-1,
        sthresh=15,
        mthresh=11,
        close=2,
        filter_params={'a_t': 50, 'a_h': 8, 'max_n_holes': 10}
    )

    # Create heatmap
    heatmap = draw_heatmap(
        scores=attention,
        coords=coords,
        slide_path=slide_path,
        wsi_object=wsi,
        vis_level=vis_level,
        patch_size=(patch_size, patch_size),
        cmap=cmap,
        alpha=alpha,
        convert_to_percentiles=True,
        segment=False,  # Disable tissue masking for now - can enable later
        blank_canvas=False,
        blur=False
    )

    # Save overlay version
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    heatmap.save(output_path)
    print(f"  Saved overlay to: {output_path}")

    # Also save pure heatmap on blank canvas for debugging
    pure_output = output_path.replace('_heatmap.png', '_heatmap_pure.png')
    pure_heatmap = draw_heatmap(
        scores=attention,
        coords=coords,
        slide_path=slide_path,
        wsi_object=wsi,
        vis_level=vis_level,
        patch_size=(patch_size, patch_size),
        cmap=cmap,
        alpha=1.0,  # Full opacity
        convert_to_percentiles=True,
        segment=False,
        blank_canvas=True,  # White background
        blur=False
    )
    pure_heatmap.save(pure_output)
    print(f"  Saved pure heatmap to: {pure_output}")

    return {
        'slide_id': Path(slide_path).stem,
        'num_patches': len(features),
        'prediction': pred,
        'probabilities': probs.tolist() if probs is not None else None,
        'attention_mean': float(attention.mean()),
        'attention_std': float(attention.std()),
        'heatmap_path': output_path,
        'pure_heatmap_path': pure_output
    }


def main():
    parser = argparse.ArgumentParser(description='Generate attention heatmaps')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--h5_dir', type=str, required=True,
                       help='Directory containing H5 embedding files')
    parser.add_argument('--slide_dir', type=str, required=True,
                       help='Directory containing WSI files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for heatmaps')

    # Model config
    parser.add_argument('--model_name', type=str, default='abmil.base.uni_v2.pc108-24k',
                       help='Model name (default: abmil.base.uni_v2.pc108-24k)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (default: 2)')

    # Visualization
    parser.add_argument('--cmap', type=str, default='jet',
                       help='Colormap: jet, coolwarm, viridis, plasma (default: jet)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency 0-1 (default: 0.5)')
    parser.add_argument('--vis_level', type=int, default=-1,
                       help='Visualization level, -1=auto (default: -1)')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size (default: 256)')

    # File selection
    parser.add_argument('--slide_ext', type=str, default='.tiff',
                       help='Slide extension (default: .tiff)')
    parser.add_argument('--csv', type=str, default=None,
                       help='CSV with slide_id column to process specific slides')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of slides to process')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")

    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint
    ).to(device)

    print("Model loaded successfully!\n")

    # Get slide list
    if args.csv:
        df = pd.read_csv(args.csv)
        slide_ids = df['slide_id'].tolist()
        print(f"Processing {len(slide_ids)} slides from CSV")
    else:
        # Find all H5 files
        h5_files = list(Path(args.h5_dir).glob('*.h5'))
        slide_ids = [f.stem for f in h5_files]
        print(f"Found {len(slide_ids)} H5 files")

    if args.limit:
        slide_ids = slide_ids[:args.limit]
        print(f"Limited to first {args.limit} slides")

    # Process slides
    results = []
    print(f"\nProcessing {len(slide_ids)} slides...")
    print("="*70)

    for slide_id in tqdm(slide_ids):
        try:
            # Build paths
            h5_path = os.path.join(args.h5_dir, f'{slide_id}.h5')
            slide_path = os.path.join(args.slide_dir, f'{slide_id}{args.slide_ext}')
            output_path = os.path.join(args.output_dir, f'{slide_id}_heatmap.png')

            # Check files exist
            if not os.path.exists(h5_path):
                print(f"  ⚠ H5 not found: {slide_id}")
                continue
            if not os.path.exists(slide_path):
                print(f"  ⚠ Slide not found: {slide_id}")
                continue

            # Generate heatmap
            print(f"\n{slide_id}")
            result = generate_heatmap(
                slide_path=slide_path,
                h5_path=h5_path,
                model=model,
                output_path=output_path,
                device=device,
                cmap=args.cmap,
                alpha=args.alpha,
                vis_level=args.vis_level,
                patch_size=args.patch_size
            )
            results.append(result)

        except Exception as e:
            print(f"  ✗ Error processing {slide_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save results summary
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)

    if results:
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.output_dir, 'heatmap_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nProcessed {len(results)} slides")
        print(f"Results saved to: {results_path}")
        print(f"Heatmaps saved to: {args.output_dir}")
    else:
        print("\nNo slides were successfully processed!")


if __name__ == '__main__':
    main()
