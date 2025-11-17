#!/usr/bin/env python3
"""
Data loading utilities for MIL training
Handles both CLAM and ABMIL datasets with support for 2D/3D embedding formats
"""

import pandas as pd
import numpy as np
import os
import torch
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob


def preprocess_panda_data(
    csv_path,
    feats_path,
    grade_group=False,
    exclude_mid_grade=False,
    min_patches=24,
    test_split=0.10,
    val_split=0.10,
    seed=10
):
    """
    Preprocess PANDA dataset: read CSV, match features, perform stratified split

    Parameters
    ----------
    csv_path : str
        Path to PANDA train.csv file
    feats_path : str
        Path to directory containing h5 feature files
    grade_group : bool
        Whether to group ISUP grades into clinical categories
    exclude_mid_grade : bool
        Whether to exclude mid grade (ISUP 2-3) - only if grade_group=True
    min_patches : int
        Minimum number of patches required to include slide (default: 24)
    test_split : float
        Fraction of data to use for test set (default: 0.10 = 10%)
    val_split : float
        Fraction of data to use for validation set (default: 0.20 = 20%)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: slide_id, label, isup_grade, split
    num_classes : int
        Number of classes
    group_names : list
        Names of the classes
    """

    print("="*70)
    print("DATA PREPROCESSING")
    if grade_group:
        if exclude_mid_grade:
            print("MODE: Clinical Grade Grouping (3 classes, mid grade excluded)")
        else:
            print("MODE: Clinical Grade Grouping (4 classes)")
    else:
        print("MODE: Original ISUP Grading (6 classes)")
    print("="*70 + "\n")

    # Set random seed
    np.random.seed(seed)

    # Step 1: Read labels from CSV
    print("Step 1: Reading labels from CSV...")
    df_labels = pd.read_csv(csv_path)[['slide_id', 'label']]
    df_labels['isup_grade'] = df_labels['label']

    # Apply grade grouping if enabled
    if grade_group:
        if exclude_mid_grade:
            print(f"  Excluding mid grade (ISUP 2-3) from analysis...")
            df_labels = df_labels[~df_labels['isup_grade'].isin([2, 3])].reset_index(drop=True)
            print(f"  Remaining slides after exclusion: {len(df_labels)}")

        def map_isup_to_group(isup_grade):
            """Map ISUP grades to clinical groups"""
            if isup_grade == 0:
                return 0  # No cancer
            elif isup_grade == 1:
                return 1  # Low grade
            elif isup_grade in [2, 3]:
                return 2  # Mid grade
            elif isup_grade in [4, 5]:
                return 2 if exclude_mid_grade else 3  # High grade
            else:
                raise ValueError(f"Invalid ISUP grade: {isup_grade}")

        df_labels['label'] = df_labels['isup_grade'].apply(map_isup_to_group)
        print(f"  Applied grade grouping: ISUP -> Clinical Groups")

    print(f"  Found {len(df_labels)} slides in CSV with labels")

    # Define group names
    if grade_group:
        if exclude_mid_grade:
            group_names = ['Group 0 (No cancer)', 'Group 1 (Low grade)', 'Group 2 (High grade)']
        else:
            group_names = ['Group 0 (No cancer)', 'Group 1 (Low grade)',
                           'Group 2 (Mid grade)', 'Group 3 (High grade)']
    else:
        group_names = [f'ISUP {i}' for i in range(6)]

    # Step 2: Find all available feature files
    print(f"\nStep 2: Scanning features directory...")
    feature_files = glob(os.path.join(feats_path, '*.h5'))
    available_slide_ids = [os.path.basename(f).replace('.h5', '') for f in feature_files]
    print(f"  Found {len(available_slide_ids)} feature files")

    # Step 3: Match CSV with available features
    print(f"\nStep 3: Matching CSV labels with available features...")
    df_labels['has_features'] = df_labels['slide_id'].isin(available_slide_ids)
    df_matched = df_labels[df_labels['has_features']].drop(columns=['has_features']).reset_index(drop=True)
    missing_count = len(df_labels) - len(df_matched)
    print(f"  Matched: {len(df_matched)} slides")
    print(f"  Missing features: {missing_count} slides")

    # Step 3.5: Filter slides with insufficient patches
    if min_patches > 0:
        print(f"\nStep 3.5: Filtering slides with < {min_patches} patches...")

        # Check if QC results exist
        qc_csv_path = 'low_patch_slides.csv'
        if os.path.exists(qc_csv_path):
            print(f"  Using existing QC results from {qc_csv_path}")
            df_qc = pd.read_csv(qc_csv_path)
            slides_to_exclude = set(df_qc['slide_id'].tolist())
            df_filtered = df_matched[~df_matched['slide_id'].isin(slides_to_exclude)].reset_index(drop=True)
            num_excluded = len(df_matched) - len(df_filtered)
            print(f"  Slides with >= {min_patches} patches: {len(df_filtered)}")
            print(f"  Slides excluded (low patch count): {num_excluded}")
        else:
            print(f"  QC results not found, filtering on-the-fly (this may take a while)...")
            slides_to_keep = []
            low_patch_count = 0

            for _, row in df_matched.iterrows():
                slide_id = row['slide_id']
                feat_path = os.path.join(feats_path, slide_id + '.h5')

                try:
                    with h5py.File(feat_path, 'r') as f:
                        features = f['features'][:]
                        # Handle both 2D (new Trident) and 3D (old) formats
                        if len(features.shape) == 3:
                            num_patches = features.shape[1]  # (1, num_patches, dim)
                        else:
                            num_patches = features.shape[0]  # (num_patches, dim)

                        if num_patches >= min_patches:
                            slides_to_keep.append(slide_id)
                        else:
                            low_patch_count += 1
                except Exception as e:
                    print(f"  [WARNING] Failed to read {slide_id}: {e}")
                    low_patch_count += 1

            df_filtered = df_matched[df_matched['slide_id'].isin(slides_to_keep)].reset_index(drop=True)
            print(f"  Slides with >= {min_patches} patches: {len(df_filtered)}")
            print(f"  Slides excluded (low patch count): {low_patch_count}")
    else:
        df_filtered = df_matched

    # Step 4: Perform stratified train/val/test split
    train_split = 1.0 - test_split - val_split
    print(f"\nStep 4: Performing stratified split ({train_split*100:.0f}% train, {val_split*100:.0f}% val, {test_split*100:.0f}% test)...")

    # First split: separate out test set
    train_val_df, test_df = train_test_split(
        df_filtered, test_size=test_split, stratify=df_filtered['label'], random_state=seed
    )

    # Second split: separate train and val from remaining data
    # val_split_adjusted is the fraction of train_val that should go to val
    val_split_adjusted = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_split_adjusted, stratify=train_val_df['label'], random_state=seed
    )

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total slides: {len(df)}\n")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    num_classes = len(df['label'].unique())
    print(f"\nNumber of classes: {num_classes}")
    print(f"{'='*70}\n")

    return df, num_classes, group_names


class PANDAH5Dataset(Dataset):
    """
    Universal dataset for PANDA with UNI v2 features
    Supports both CLAM (batch_size=1, all patches) and ABMIL (batch_size>1, sampled patches)
    Handles both 2D (new Trident) and 3D (old) embedding formats
    """

    def __init__(self, feats_path, df, split, num_features=None, seed=42):
        """
        Parameters
        ----------
        feats_path : str
            Path to directory containing h5 feature files
        df : pd.DataFrame
            DataFrame with slide_id, label, split columns
        split : str
            One of 'train', 'val', 'test'
        num_features : int, optional
            Number of patches to sample for training. If None, use all patches (for CLAM)
        seed : int
            Random seed for patch sampling
        """
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = os.path.join(self.feats_path, row['slide_id'] + '.h5')

        with h5py.File(feat_path, "r") as f:
            # Load features and handle both 2D (new Trident) and 3D (old) formats
            raw_features = torch.from_numpy(f["features"][:])

            # Trident generates 2D: (num_patches, 1536)
            # Old format was 3D: (1, num_patches, 1536)
            # Use .squeeze(0).clone() for both formats:
            # - 3D (1, M, D) -> squeeze -> 2D (M, D) -> clone for resizable storage
            # - 2D (M, D) -> squeeze(0) is no-op -> clone for resizable storage
            # .clone() is needed because h5py creates non-resizable storage
            features = raw_features.squeeze(0).clone()

        # Sample patches for training if num_features specified (ABMIL)
        if self.num_features is not None and self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(self.seed))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(self.seed))
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


def create_dataloaders(
    feats_path,
    df,
    batch_size=1,
    num_features=None,
    num_workers=4,
    weighted_sampling=True,
    seed=42
):
    """
    Create train/val/test dataloaders

    Parameters
    ----------
    feats_path : str
        Path to feature directory
    df : pd.DataFrame
        Preprocessed dataframe with split column
    batch_size : int
        Batch size (1 for CLAM, >1 for ABMIL)
    num_features : int, optional
        Number of patches to sample for training (ABMIL only)
    num_workers : int
        Number of dataloader workers
    weighted_sampling : bool
        Use weighted sampling for class balancing (default: True)
    seed : int
        Random seed

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """

    train_dataset = PANDAH5Dataset(feats_path, df, "train", num_features=num_features, seed=seed)
    val_dataset = PANDAH5Dataset(feats_path, df, "val", num_features=num_features, seed=seed)
    test_dataset = PANDAH5Dataset(feats_path, df, "test", num_features=num_features, seed=seed)

    # Create weighted sampler if requested
    if weighted_sampling:
        from torch.utils.data import WeightedRandomSampler
        train_df = df[df['split'] == 'train']
        train_labels = train_df['label'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]

        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=num_workers
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
