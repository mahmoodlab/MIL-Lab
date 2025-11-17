#!/usr/bin/env python3
"""
Script to check patch counts in h5 feature files and report slides with insufficient patches.
Useful for quality control to identify slides that may need reprocessing.
"""

import h5py
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import argparse


def check_patch_counts(feats_path, min_patches=24, csv_path=None, output_csv=None):
    """
    Check all h5 files and report those with fewer than min_patches.

    Parameters
    ----------
    feats_path : str
        Path to directory containing h5 feature files
    min_patches : int
        Minimum number of patches required (default: 24)
    csv_path : str, optional
        Path to CSV with labels to include label information
    output_csv : str, optional
        Path to save results CSV

    Returns
    -------
    pd.DataFrame
        DataFrame with slide_id, num_patches, and optionally label
    """

    print(f"{'='*70}")
    print(f"PATCH COUNT QUALITY CONTROL")
    print(f"{'='*70}")
    print(f"Features directory: {feats_path}")
    print(f"Minimum patches threshold: {min_patches}")
    print(f"{'='*70}\n")

    # Load labels if provided
    labels_dict = {}
    if csv_path:
        print(f"Loading labels from: {csv_path}")
        df_labels = pd.read_csv(csv_path)[['slide_id', 'label']]
        labels_dict = dict(zip(df_labels['slide_id'], df_labels['label']))
        print(f"  Found {len(labels_dict)} slides with labels\n")

    # Find all h5 files
    print("Scanning for h5 files...")
    feature_files = glob(os.path.join(feats_path, '*.h5'))
    print(f"  Found {len(feature_files)} h5 files\n")

    # Check patch counts
    print(f"Checking patch counts (threshold: {min_patches})...\n")

    results = []
    low_patch_count = 0

    for feat_file in tqdm(feature_files, desc="Processing"):
        slide_id = os.path.basename(feat_file).replace('.h5', '')

        try:
            with h5py.File(feat_file, 'r') as f:
                features = f['features'][:]

                # Handle both 2D (new Trident) and 3D (old) formats
                if len(features.shape) == 3:
                    num_patches = features.shape[1]  # (1, num_patches, dim)
                else:
                    num_patches = features.shape[0]  # (num_patches, dim)

                # Get label if available
                label = labels_dict.get(slide_id, None)

                # Record if below threshold
                if num_patches < min_patches:
                    results.append({
                        'slide_id': slide_id,
                        'num_patches': num_patches,
                        'label': label,
                        'file_path': feat_file
                    })
                    low_patch_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to read {slide_id}: {e}")
            results.append({
                'slide_id': slide_id,
                'num_patches': -1,
                'label': None,
                'file_path': feat_file,
                'error': str(e)
            })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total slides scanned:           {len(feature_files)}")
    print(f"Slides below threshold:         {low_patch_count} ({low_patch_count/len(feature_files)*100:.2f}%)")
    print(f"Slides meeting threshold:       {len(feature_files) - low_patch_count}")

    if len(df_results) > 0:
        print(f"\nPatch count distribution for low-count slides:")
        print(f"  Min patches:     {df_results['num_patches'].min()}")
        print(f"  Max patches:     {df_results['num_patches'].max()}")
        print(f"  Mean patches:    {df_results['num_patches'].mean():.1f}")
        print(f"  Median patches:  {df_results['num_patches'].median():.1f}")

        # Show patch count histogram
        print(f"\nPatch count histogram:")
        bins = [0, 5, 10, 15, 20, min_patches]
        for i in range(len(bins)-1):
            count = len(df_results[(df_results['num_patches'] >= bins[i]) &
                                   (df_results['num_patches'] < bins[i+1])])
            print(f"  {bins[i]:2d}-{bins[i+1]-1:2d} patches: {count:4d} slides")

        # Label distribution if available
        if csv_path and 'label' in df_results.columns:
            print(f"\nLabel distribution for low-count slides:")
            label_counts = df_results['label'].value_counts().sort_index()
            for label, count in label_counts.items():
                if pd.notna(label):
                    print(f"  Label {int(label)}: {count:4d} slides")
            unlabeled = df_results['label'].isna().sum()
            if unlabeled > 0:
                print(f"  No label: {unlabeled:4d} slides")

    print(f"{'='*70}\n")

    # Save to CSV if requested
    if output_csv and len(df_results) > 0:
        df_results = df_results.sort_values('num_patches')
        df_results.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
        print(f"  Columns: {list(df_results.columns)}")
        print(f"  Total rows: {len(df_results)}\n")

        # Show first 10 slides with lowest patch counts
        print("Top 10 slides with lowest patch counts:")
        print(df_results[['slide_id', 'num_patches', 'label']].head(10).to_string(index=False))
        print()

    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Check patch counts in h5 feature files for quality control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - check for slides with < 24 patches
  python check_patch_counts.py --feats-path /path/to/features/

  # Custom threshold
  python check_patch_counts.py --feats-path /path/to/features/ --min-patches 50

  # Include labels from CSV
  python check_patch_counts.py --feats-path /path/to/features/ \\
      --csv-path /path/to/train.csv

  # Save results to CSV
  python check_patch_counts.py --feats-path /path/to/features/ \\
      --min-patches 24 --output low_patch_slides.csv
        """
    )

    parser.add_argument('--feats-path', type=str, required=True,
                        help='Path to directory containing h5 feature files')
    parser.add_argument('--min-patches', type=int, default=24,
                        help='Minimum number of patches required (default: 24)')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Path to CSV with slide labels (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results CSV (optional)')

    args = parser.parse_args()

    # Run check
    df_results = check_patch_counts(
        feats_path=args.feats_path,
        min_patches=args.min_patches,
        csv_path=args.csv_path,
        output_csv=args.output
    )

    # Exit code based on results
    if len(df_results) > 0:
        print(f"⚠️  Found {len(df_results)} slides with < {args.min_patches} patches")
        print(f"   Review these slides for potential quality issues.\n")
        return 1
    else:
        print(f"✓ All slides have >= {args.min_patches} patches\n")
        return 0


if __name__ == "__main__":
    exit(main())
