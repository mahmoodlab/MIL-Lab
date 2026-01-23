#!/usr/bin/env python3
"""
Prepare PANDA data for the new training infrastructure.

Converts the PathBench TSV format to the labels CSV format expected by MILDataset.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


def prepare_panda_labels(
    tsv_path: str,
    features_dir: str,
    output_csv: str,
    fold_column: str = 'fold_0',
    val_fraction: float = 0.1,
    min_patches: int = 24,
    seed: int = 10,
):
    """
    Convert PANDA TSV to labels CSV format.

    Args:
        tsv_path: Path to k=all.tsv
        features_dir: Path to features directory (to match available slides)
        output_csv: Output path for labels CSV
        fold_column: Column containing train/test split
        val_fraction: Fraction of training data to use for validation
        min_patches: Minimum number of patches required (default: 24)
        seed: Random seed
    """
    print("=" * 70)
    print("PREPARING PANDA DATA")
    print("=" * 70 + "\n")

    # Load TSV
    print(f"Loading TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    print(f"  Total slides in TSV: {len(df)}")

    # Rename columns for compatibility
    df = df.rename(columns={'isup_grade': 'label'})
    df['label'] = df['label'].astype(str)  # MILDataset expects string labels

    # Get split from fold column
    df['split'] = df[fold_column]

    # Find available features
    print(f"\nScanning features directory: {features_dir}")
    features_dir = Path(features_dir)
    available_slides = set(p.stem for p in features_dir.glob('*.h5'))
    print(f"  Available feature files: {len(available_slides)}")

    # Match with available features
    df = df[df['slide_id'].isin(available_slides)].reset_index(drop=True)
    print(f"  Matched slides: {len(df)}")

    # Filter by min_patches
    if min_patches > 0:
        print(f"\nFiltering slides with < {min_patches} patches...")
        slides_to_keep = []
        low_patch_count = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking patch counts"):
            slide_id = row['slide_id']
            feat_path = features_dir / f"{slide_id}.h5"

            try:
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]
                    # Handle both 2D (num_patches, dim) and 3D (1, num_patches, dim) formats
                    if len(features.shape) == 3:
                        num_patches = features.shape[1]
                    else:
                        num_patches = features.shape[0]

                    if num_patches >= min_patches:
                        slides_to_keep.append(slide_id)
                    else:
                        low_patch_count += 1
            except Exception as e:
                print(f"  [WARNING] Failed to read {slide_id}: {e}")
                low_patch_count += 1

        df = df[df['slide_id'].isin(slides_to_keep)].reset_index(drop=True)
        print(f"  Slides with >= {min_patches} patches: {len(df)}")
        print(f"  Slides excluded (low patch count): {low_patch_count}")

    # Create validation split from training data
    if val_fraction > 0:
        print(f"\nCreating validation split ({val_fraction*100:.0f}% of training)...")
        np.random.seed(seed)

        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)

        train_df_new, val_df = train_test_split(
            train_df,
            test_size=val_fraction,
            stratify=train_df['label'],
            random_state=seed,
        )

        train_df_new['split'] = 'train'
        val_df['split'] = 'val'

        df = pd.concat([train_df_new, val_df, test_df], ignore_index=True)

    # Print summary
    print(f"\n{'=' * 70}")
    print("SPLIT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total slides: {len(df)}\n")
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]
        if len(split_df) > 0:
            print(f"  {split_name.capitalize():5s}: {len(split_df)} ({len(split_df)/len(df)*100:.1f}%)")

    print(f"\nLabel distribution:")
    for label in sorted(df['label'].unique()):
        count = len(df[df['label'] == label])
        print(f"  ISUP {label}: {count}")

    # Select columns for output
    output_df = df[['slide_id', 'label', 'split']].copy()
    if 'case_id' in df.columns:
        output_df['case_id'] = df['case_id']

    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"\n{'=' * 70}")
    print(f"Saved to: {output_csv}")
    print(f"{'=' * 70}\n")

    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare PANDA data')
    parser.add_argument('--tsv', type=str, required=True, help='Path to k=all.tsv')
    parser.add_argument('--features', type=str, required=True, help='Path to features dir')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--fold-column', type=str, default='fold_0', help='Fold column name')
    parser.add_argument('--val-fraction', type=float, default=0.1, help='Validation fraction')
    parser.add_argument('--min-patches', type=int, default=24, help='Minimum patches per slide (default: 24)')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')

    args = parser.parse_args()

    prepare_panda_labels(
        tsv_path=args.tsv,
        features_dir=args.features,
        output_csv=args.output,
        fold_column=args.fold_column,
        val_fraction=args.val_fraction,
        min_patches=args.min_patches,
        seed=args.seed,
    )
