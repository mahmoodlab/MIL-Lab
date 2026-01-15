#!/usr/bin/env python3
"""
Debug heatmap generation - check what's going wrong

Usage:
    python debug_heatmap.py \
        --checkpoint path/to/model.pt \
        --h5_file path/to/slide.h5 \
        --slide path/to/slide.tiff
"""

import sys
from pathlib import Path

# Add grandparent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.builder import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--h5_file', type=str, required=True)
    parser.add_argument('--slide', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='abmil.base.uni_v2.pc108-24k')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Step 1: Load H5 file
    print("="*70)
    print("STEP 1: Loading H5 file")
    print("="*70)
    print(f"H5 file: {args.h5_file}")

    with h5py.File(args.h5_file, 'r') as f:
        print(f"\nH5 file contents: {list(f.keys())}")

        # Features
        raw_features = f['features'][:]
        print(f"\nFeatures:")
        print(f"  Raw shape: {raw_features.shape}")
        print(f"  Raw dtype: {raw_features.dtype}")

        features = torch.from_numpy(raw_features).squeeze(0).clone()
        print(f"  After squeeze: {features.shape}")

        # Coordinates
        coords = f['coords'][:]
        print(f"\nCoordinates:")
        print(f"  Shape: {coords.shape}")
        print(f"  dtype: {coords.dtype}")
        print(f"  Min: {coords.min(axis=0)}")
        print(f"  Max: {coords.max(axis=0)}")
        print(f"  First 5: {coords[:5]}")

    # Step 2: Load model
    print("\n" + "="*70)
    print("STEP 2: Loading model")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model name: {args.model_name}")

    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint
    ).to(device)

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Model device: {next(model.parameters()).device}")

    # Step 3: Run inference
    print("\n" + "="*70)
    print("STEP 3: Running inference")
    print("="*70)

    model.eval()
    features_batch = features.unsqueeze(0).to(device)
    print(f"Input shape: {features_batch.shape}")

    with torch.inference_mode():
        # Try to get attention
        print("\nCalling model with return_attention=True...")
        try:
            results_dict, log_dict = model(features_batch, return_attention=True)

            print(f"\nResults dict keys: {results_dict.keys()}")
            print(f"Log dict keys: {log_dict.keys()}")

            # Check for attention
            if 'attention' in log_dict and log_dict['attention'] is not None:
                attention_raw = log_dict['attention']
                print(f"\n✓ Attention found!")
                print(f"  Raw shape: {attention_raw.shape}")
                print(f"  Device: {attention_raw.device}")
                print(f"  dtype: {attention_raw.dtype}")

                # Apply softmax (ABMIL returns raw scores)
                attention_softmax = torch.softmax(attention_raw, dim=-1)
                attention_np = attention_softmax.cpu().numpy().squeeze()

                print(f"  After softmax + squeeze: {attention_np.shape}")
                print(f"  Min: {attention_np.min():.6f}")
                print(f"  Max: {attention_np.max():.6f}")
                print(f"  Mean: {attention_np.mean():.6f}")
                print(f"  Std: {attention_np.std():.6f}")
                print(f"  Sum (should be ~1.0): {attention_np.sum():.6f}")
                print(f"  Non-zero: {(attention_np > 1e-6).sum()} / {len(attention_np)}")

                # Show distribution
                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.hist(attention_np, bins=50)
                plt.title('Attention Distribution')
                plt.xlabel('Attention Score')
                plt.ylabel('Count')

                plt.subplot(1, 3, 2)
                plt.scatter(coords[:, 0], coords[:, 1], c=attention_np, s=1, cmap='jet')
                plt.colorbar(label='Attention')
                plt.title('Attention Spatial Distribution')
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')

                plt.subplot(1, 3, 3)
                sorted_attn = np.sort(attention_np)
                plt.plot(sorted_attn)
                plt.title('Sorted Attention Scores')
                plt.xlabel('Patch index (sorted)')
                plt.ylabel('Attention')

                plt.tight_layout()
                plt.savefig('debug_attention.png', dpi=150, bbox_inches='tight')
                print(f"\n✓ Saved attention visualization to: debug_attention.png")

            else:
                print(f"\n✗ No 'attention' in log_dict!")
                print(f"Available keys: {log_dict.keys()}")

            # Check predictions
            if 'Y_hat' in results_dict:
                Y_hat = results_dict['Y_hat']
                print(f"\nPrediction: {Y_hat.item()}")

            if 'Y_prob' in results_dict:
                Y_prob = results_dict['Y_prob']
                print(f"Probabilities: {Y_prob.cpu().numpy()}")

        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()

    # Step 4: Check slide
    print("\n" + "="*70)
    print("STEP 4: Checking slide")
    print("="*70)
    print(f"Slide path: {args.slide}")

    try:
        import openslide
        wsi = openslide.open_slide(args.slide)
        print(f"✓ Slide opened successfully")
        print(f"  Dimensions (level 0): {wsi.dimensions}")
        print(f"  Number of levels: {wsi.level_count}")
        print(f"  Level dimensions: {wsi.level_dimensions}")
        print(f"  Downsamples: {wsi.level_downsamples}")
    except Exception as e:
        print(f"✗ Error opening slide: {e}")

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    print("\nCheck debug_attention.png to see attention distribution")


if __name__ == '__main__':
    main()
