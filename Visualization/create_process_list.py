"""
Create Process List for Heatmap Generation

This script helps create a CSV file listing slides to process for heatmap generation.

Usage:
    # Basic usage
    python create_process_list.py --slide_dir /path/to/slides --output my_list.csv

    # With features directory
    python create_process_list.py \
        --slide_dir /path/to/slides \
        --features_dir features/h5_files \
        --output my_list.csv

    # With labels from CSV
    python create_process_list.py \
        --slide_dir /path/to/slides \
        --labels_csv /path/to/labels.csv \
        --output my_list.csv
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def create_process_list(
    slide_dir,
    features_dir=None,
    labels_csv=None,
    slide_ext='.svs',
    output_path='heatmaps/process_lists/process_list.csv',
    check_features=True
):
    """
    Create process list CSV from slide directory

    Args:
        slide_dir: Directory containing WSI files
        features_dir: Directory containing pre-extracted features
        labels_csv: CSV with slide labels (columns: slide_id, label)
        slide_ext: Slide file extension
        output_path: Output CSV path
        check_features: Whether to verify feature files exist

    Returns:
        DataFrame with process list
    """
    print(f"Scanning slide directory: {slide_dir}")

    # Find all slides
    if not os.path.exists(slide_dir):
        raise ValueError(f"Slide directory not found: {slide_dir}")

    slides = [f for f in os.listdir(slide_dir) if f.endswith(slide_ext)]
    slide_ids = [f.replace(slide_ext, '') for f in slides]

    print(f"Found {len(slides)} slides with extension '{slide_ext}'")

    if len(slides) == 0:
        print(f"Warning: No slides found with extension '{slide_ext}'")
        print(f"Available extensions: {set([os.path.splitext(f)[1] for f in os.listdir(slide_dir)])}")
        return None

    # Create base dataframe
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'process': 1
    })

    # Add labels if provided
    if labels_csv is not None:
        if os.path.exists(labels_csv):
            print(f"Loading labels from: {labels_csv}")
            labels_df = pd.read_csv(labels_csv)

            # Merge on slide_id
            if 'slide_id' in labels_df.columns and 'label' in labels_df.columns:
                df = df.merge(labels_df[['slide_id', 'label']], on='slide_id', how='left')
                print(f"  Matched {df['label'].notna().sum()} slides with labels")
            else:
                print(f"Warning: labels CSV must have 'slide_id' and 'label' columns")
        else:
            print(f"Warning: Labels CSV not found: {labels_csv}")

    # Add feature paths
    if features_dir is not None:
        print(f"Looking for features in: {features_dir}")

        features_paths = []
        coords_paths = []
        missing_features = []

        for slide_id in slide_ids:
            # Feature file (.pt)
            feat_path = os.path.join(features_dir, f'{slide_id}.pt')
            if os.path.exists(feat_path):
                features_paths.append(feat_path)
            else:
                features_paths.append('')
                if check_features:
                    missing_features.append(slide_id)

            # Coordinates file (.h5)
            coord_path = os.path.join(features_dir, f'{slide_id}.h5')
            if os.path.exists(coord_path):
                coords_paths.append(coord_path)
            else:
                coords_paths.append('')

        df['features_path'] = features_paths
        df['coords_path'] = coords_paths

        # Report missing features
        if missing_features:
            print(f"\nWarning: {len(missing_features)} slides missing feature files:")
            for sid in missing_features[:10]:
                print(f"  - {sid}")
            if len(missing_features) > 10:
                print(f"  ... and {len(missing_features) - 10} more")

            if check_features:
                # Mark slides without features as process=0
                df.loc[df['features_path'] == '', 'process'] = 0
                print(f"\nSet process=0 for {len(missing_features)} slides without features")

        print(f"Found features for {(df['features_path'] != '').sum()} slides")
        print(f"Found coordinates for {(df['coords_path'] != '').sum()} slides")

    # Summary
    print(f"\nProcess List Summary:")
    print(f"  Total slides: {len(df)}")
    print(f"  To process (process=1): {(df['process'] == 1).sum()}")
    print(f"  To skip (process=0): {(df['process'] == 0).sum()}")

    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        print(df[df['process'] == 1]['label'].value_counts())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved process list to: {output_path}")

    # Show sample
    print(f"\nFirst 5 rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head())

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create process list CSV for heatmap generation'
    )
    parser.add_argument('--slide_dir', type=str, required=True,
                       help='Directory containing WSI files')
    parser.add_argument('--features_dir', type=str, default=None,
                       help='Directory containing extracted features')
    parser.add_argument('--labels_csv', type=str, default=None,
                       help='CSV with slide labels (columns: slide_id, label)')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                       help='Slide file extension (default: .svs)')
    parser.add_argument('--output', type=str, default='heatmaps/process_lists/process_list.csv',
                       help='Output CSV path')
    parser.add_argument('--no_check_features', action='store_true',
                       help='Do not verify feature files exist')

    args = parser.parse_args()

    create_process_list(
        slide_dir=args.slide_dir,
        features_dir=args.features_dir,
        labels_csv=args.labels_csv,
        slide_ext=args.slide_ext,
        output_path=args.output,
        check_features=not args.no_check_features
    )


if __name__ == '__main__':
    main()
