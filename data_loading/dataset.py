#!/usr/bin/env python3
"""
Dataset module for MIL training.

Supports multiple levels of aggregation for Multiple Instance Learning:

    1. Slide-level (MILDataset):
       - One bag per H5 file
       - Use for: PANDA (1 patient = 1 slide)

    2. Grouped/Concatenated (GroupedMILDataset via .concat_by()):
       - Concatenates multiple slides into one bag
       - Use for: Multi-slide patients where slide boundaries don't matter

    3. Hierarchical (HierarchicalMILDataset via .group_by()) [FUTURE]:
       - Preserves slide structure within patient
       - Use for: When you need slide-level attention THEN patient-level attention

Design:
    - Each data source is loaded independently
    - Joined on slide_id when needed
    - Grouping is done via methods, not separate classes for each level
"""

import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional, Dict, Union
from dataclasses import dataclass, field

from .feature_loader import load_features, get_slide_ids, get_embed_dim


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SlideData:
    """Container for a single slide's data (slide-level MIL)."""
    slide_id: str
    features: torch.Tensor  # [M, D] - M patches, D dimensions
    label: Optional[str] = None
    case_id: Optional[str] = None


@dataclass
class GroupedData:
    """Container for grouped data (e.g., patient-level with concatenated slides)."""
    group_id: str
    features: torch.Tensor  # [M_total, D] - concatenated from all items in group
    label: Optional[str] = None
    item_ids: Optional[List[str]] = None  # e.g., slide_ids in the group
    num_items: int = 1


@dataclass
class HierarchicalData:
    """
    Container for hierarchical data (preserves structure).

    Use when you need two-stage attention:
    1. Patch-level attention within each slide
    2. Slide-level attention across slides
    """
    group_id: str
    features: List[torch.Tensor]  # List of [M_i, D] tensors, one per slide
    label: Optional[str] = None
    item_ids: Optional[List[str]] = None
    core_ids: Optional[List[str]] = None
    num_items: int = 1

    def to_padded_tensor(self) -> tuple:
        """
        Convert to padded tensor for batching.

        Returns:
            features: [num_items, max_patches, D]
            mask: [num_items, max_patches] - 1 for valid, 0 for padding
        """
        max_patches = max(f.shape[0] for f in self.features)
        embed_dim = self.features[0].shape[1]

        padded = torch.zeros(self.num_items, max_patches, embed_dim)
        mask = torch.zeros(self.num_items, max_patches)

        for i, feat in enumerate(self.features):
            n = feat.shape[0]
            padded[i, :n, :] = feat
            mask[i, :n] = 1.0

        return padded, mask


# =============================================================================
# Labels (CSV)
# =============================================================================

def load_labels(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load labels from CSV.

    Expected columns: slide_id, label
    Optional columns: case_id, split, etc.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with at least: slide_id, label
    """
    df = pd.read_csv(csv_path)

    required = ['slide_id', 'label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure slide_id and label are strings
    df['slide_id'] = df['slide_id'].astype(str)
    df['label'] = df['label'].astype(str)

    return df


# =============================================================================
# Features (H5 Directory)
# =============================================================================

def get_available_features(features_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Get metadata about available feature files.

    Args:
        features_dir: Directory containing .h5 files

    Returns:
        DataFrame with: slide_id, h5_path
    """
    features_dir = Path(features_dir)
    h5_files = list(features_dir.glob('*.h5'))

    return pd.DataFrame({
        'slide_id': [f.stem for f in h5_files],
        'h5_path': [str(f) for f in h5_files],
    })


# =============================================================================
# Join: Labels + Features
# =============================================================================

def join_labels_and_features(
    labels_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join labels with available features.

    Args:
        labels_df: DataFrame with slide_id, label columns
        features_df: DataFrame with slide_id, h5_path columns

    Returns:
        DataFrame with slides that have both labels AND features
    """
    merged = labels_df.merge(features_df, on='slide_id', how='inner')

    n_labels = len(labels_df)
    n_features = len(features_df)
    n_matched = len(merged)

    print(f"Labels:   {n_labels} slides")
    print(f"Features: {n_features} slides")
    print(f"Matched:  {n_matched} slides")

    if n_matched < n_labels:
        missing_features = set(labels_df.slide_id) - set(features_df.slide_id)
        print(f"Missing features: {len(missing_features)} slides")

    if n_matched < n_features:
        missing_labels = set(features_df.slide_id) - set(labels_df.slide_id)
        print(f"Missing labels: {len(missing_labels)} slides")

    return merged


# =============================================================================
# Slide-Level Dataset (Base)
# =============================================================================

class MILDataset:
    """
    Slide-level MIL dataset. One bag per H5 file.

    This is the base dataset for MIL. Use this when:
    - Each slide is an independent sample (e.g., PANDA: 1 patient = 1 slide)
    - You want slide-level predictions

    For multi-slide patients, use:
    - .concat_by('case_id') for flat concatenation
    - .group_by('case_id') for hierarchical (preserves slide boundaries)

    Usage:
        dataset = MILDataset('labels.csv', 'features/')

        # Iterate over slides
        for slide in dataset:
            features = slide.features  # [M, D]
            label = slide.label

        # Group by patient (flat concatenation)
        patient_dataset = dataset.concat_by('case_id')

        # Group by patient (hierarchical)
        patient_dataset = dataset.group_by('case_id')
    """

    def __init__(
        self,
        labels_csv: Union[str, Path],
        features_dir: Union[str, Path],
    ):
        self.features_dir = Path(features_dir)

        # Load and join
        labels_df = load_labels(labels_csv)
        features_df = get_available_features(features_dir)
        self.df = join_labels_and_features(labels_df, features_df)

        # Index for fast lookup
        self._index = {row.slide_id: idx for idx, row in self.df.iterrows()}

        # Compute once at init
        self.embed_dim = get_embed_dim(self.features_dir)
        self.num_classes = self.df.label.nunique()

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self):
        for _, row in self.df.iterrows():
            yield self._load_slide(row)

    def __getitem__(self, key: Union[int, str]) -> SlideData:
        if isinstance(key, int):
            row = self.df.iloc[key]
        else:
            idx = self._index[key]
            row = self.df.iloc[idx]
        return self._load_slide(row)

    def _load_slide(self, row: pd.Series) -> SlideData:
        """Load a single slide's data."""
        features = load_features(row.h5_path)
        return SlideData(
            slide_id=row.slide_id,
            features=features,
            label=row.label,
            case_id=row.get('case_id'),
        )

    @property
    def slide_ids(self) -> List[str]:
        return self.df.slide_id.tolist()

    @property
    def labels(self) -> List[str]:
        return self.df.label.tolist()

    def get_subset(self, slide_ids: List[str]) -> 'MILDataset':
        """Create a subset with specific slides."""
        subset = MILDataset.__new__(MILDataset)
        subset.features_dir = self.features_dir
        subset.df = self.df[self.df.slide_id.isin(slide_ids)].reset_index(drop=True)
        subset._index = {row.slide_id: idx for idx, row in subset.df.iterrows()}
        subset.embed_dim = self.embed_dim
        subset.num_classes = self.num_classes
        return subset

    def split_by_column(self, column: str = 'split') -> Dict[str, 'MILDataset']:
        """Split dataset by a column (e.g., 'split' with train/val/test)."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in dataset")

        result = {}
        for value in self.df[column].unique():
            slide_ids = self.df[self.df[column] == value].slide_id.tolist()
            result[value] = self.get_subset(slide_ids)

        return result

    def random_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42
    ) -> Dict[str, 'MILDataset']:
        """
        Randomly split dataset into train/val/test.

        WARNING: This splits by slide_id, not case_id. If you have multiple
        slides per patient, use .concat_by('case_id').random_split() instead
        to avoid data leakage.
        """
        import numpy as np
        np.random.seed(seed)

        n = len(self.df)
        indices = np.random.permutation(n)

        train_end = int(train_frac * n)
        val_end = train_end + int(val_frac * n)

        train_ids = self.df.iloc[indices[:train_end]].slide_id.tolist()
        val_ids = self.df.iloc[indices[train_end:val_end]].slide_id.tolist()
        test_ids = self.df.iloc[indices[val_end:]].slide_id.tolist()

        return {
            'train': self.get_subset(train_ids),
            'val': self.get_subset(val_ids),
            'test': self.get_subset(test_ids),
        }

    def concat_by(self, column: str, label_voting: str = 'max') -> 'GroupedMILDataset':
        """
        Group samples by column and concatenate features.

        Creates a new dataset where each sample is a group (e.g., patient)
        with features concatenated from all items in the group.

        Args:
            column: Column to group by (e.g., 'case_id')
            label_voting: How to determine group label when items have different labels.
                - 'max': Use maximum label (MIL convention: positive if any positive)
                - 'maj': Use majority vote (mode)
                - 'first': Use first item's label (assumes consistent labels)

        Returns:
            GroupedMILDataset with concatenated features

        Example:
            # Patient-level dataset with concatenated slides
            patient_dataset = slide_dataset.concat_by('case_id', label_voting='max')
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in dataset. "
                           f"Available: {list(self.df.columns)}")

        return GroupedMILDataset(self, group_column=column, label_voting=label_voting)

    def group_by(self, column: str, label_voting: str = 'max') -> 'HierarchicalMILDataset':
        """
        Group samples by column, preserving structure for hierarchical MIL.

        Creates a new dataset where each sample contains a LIST of feature
        tensors (one per item in the group), enabling two-stage attention.

        Args:
            column: Column to group by (e.g., 'case_id')
            label_voting: How to determine group label when items have different labels.
                - 'max': Use maximum label (MIL convention: positive if any positive)
                - 'maj': Use majority vote (mode)
                - 'first': Use first item's label (assumes consistent labels)

        Returns:
            HierarchicalMILDataset with preserved structure

        Example:
            # Patient-level with slide structure preserved
            patient_dataset = slide_dataset.group_by('case_id', label_voting='max')

            # Each sample has features as list of tensors
            patient = patient_dataset[0]
            for slide_features in patient.features:
                print(slide_features.shape)  # [M_i, D] for each slide
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not in dataset. "
                           f"Available: {list(self.df.columns)}")

        return HierarchicalMILDataset(self, group_column=column, label_voting=label_voting)


# =============================================================================
# Grouped Dataset (Flat Concatenation)
# =============================================================================

class GroupedMILDataset:
    """
    Grouped MIL dataset with flat concatenation.

    Concatenates all items (e.g., slides) in a group (e.g., patient) into
    one bag. The model sees one large bag of patches without knowing which
    slide they came from.

    Use when:
    - Slide boundaries are arbitrary (e.g., same tissue split into files)
    - You don't need slide-level attention

    Created via: MILDataset.concat_by('case_id', label_voting='max')
    """

    def __init__(self, base_dataset: MILDataset, group_column: str, label_voting: str = 'max'):
        """
        Args:
            base_dataset: The slide-level MILDataset
            group_column: Column to group by (e.g., 'case_id')
            label_voting: How to determine group label ('max', 'maj', 'first')
        """
        self.base = base_dataset
        self.group_column = group_column
        self.label_voting = label_voting
        self.features_dir = base_dataset.features_dir

        # Build grouped dataframe
        self.item_df = base_dataset.df  # Original slide-level df
        self.group_df = self._build_group_df()
        self._index = {row[group_column]: idx for idx, row in self.group_df.iterrows()}

        # Inherit from base
        self.embed_dim = base_dataset.embed_dim
        self.num_classes = self.group_df.label.nunique()

        print(f"\nGrouped by '{group_column}' (label_voting='{label_voting}'):")
        print(f"  Total items: {len(self.item_df)}")
        print(f"  Total groups: {len(self.group_df)}")
        print(f"  Avg items/group: {len(self.item_df) / len(self.group_df):.2f}")

    def _build_group_df(self) -> pd.DataFrame:
        """Aggregate items into groups."""
        # Determine label aggregation function based on voting method
        if self.label_voting == 'max':
            label_agg = 'max'
        elif self.label_voting == 'maj':
            # Mode (majority vote) - take first mode if tie
            label_agg = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        elif self.label_voting == 'first':
            label_agg = 'first'
        else:
            raise ValueError(f"Unknown label_voting: {self.label_voting}. "
                           f"Choose from: 'max', 'maj', 'first'")

        grouped = self.item_df.groupby(self.group_column).agg({
            'label': label_agg,
            'slide_id': list,
            'h5_path': list,
        }).reset_index()

        # Check for label conflicts and warn
        conflicts = self._check_label_conflicts()
        if conflicts:
            print(f"  WARNING: {len(conflicts)} groups have inconsistent labels!")
            if len(conflicts) <= 5:
                for group_id, labels in conflicts.items():
                    print(f"    {group_id}: {labels}")
            else:
                print(f"    (showing first 5)")
                for group_id, labels in list(conflicts.items())[:5]:
                    print(f"    {group_id}: {labels}")

        # Preserve split column if exists
        if 'split' in self.item_df.columns:
            split_map = self.item_df.groupby(self.group_column)['split'].first()
            grouped['split'] = grouped[self.group_column].map(split_map)

        grouped['num_items'] = grouped['slide_id'].apply(len)

        return grouped

    def _check_label_conflicts(self) -> Dict[str, List]:
        """Check for groups with inconsistent labels."""
        conflicts = {}
        for group_id, group_df in self.item_df.groupby(self.group_column):
            unique_labels = group_df['label'].unique()
            if len(unique_labels) > 1:
                conflicts[group_id] = unique_labels.tolist()
        return conflicts

    def __len__(self) -> int:
        return len(self.group_df)

    def __iter__(self):
        for _, row in self.group_df.iterrows():
            yield self._load_group(row)

    def __getitem__(self, key: Union[int, str]) -> GroupedData:
        if isinstance(key, int):
            row = self.group_df.iloc[key]
        else:
            idx = self._index[key]
            row = self.group_df.iloc[idx]
        return self._load_group(row)

    def _load_group(self, row: pd.Series) -> GroupedData:
        """Load and concatenate all items in the group."""
        all_features = []
        for h5_path in row.h5_path:
            features = load_features(h5_path)
            all_features.append(features)

        # Concatenate along patch dimension
        combined = torch.cat(all_features, dim=0)

        return GroupedData(
            group_id=row[self.group_column],
            features=combined,
            label=row.label,
            item_ids=row.slide_id,
            num_items=row.num_items,
        )

    @property
    def group_ids(self) -> List[str]:
        return self.group_df[self.group_column].tolist()

    @property
    def labels(self) -> List[str]:
        return self.group_df.label.tolist()

    def get_subset(self, group_ids: List[str]) -> 'GroupedMILDataset':
        """Create a subset with specific groups."""
        subset = GroupedMILDataset.__new__(GroupedMILDataset)
        subset.base = self.base
        subset.group_column = self.group_column
        subset.label_voting = self.label_voting
        subset.features_dir = self.features_dir
        subset.item_df = self.item_df[self.item_df[self.group_column].isin(group_ids)].reset_index(drop=True)
        subset.group_df = self.group_df[self.group_df[self.group_column].isin(group_ids)].reset_index(drop=True)
        subset._index = {row[self.group_column]: idx for idx, row in subset.group_df.iterrows()}
        subset.embed_dim = self.embed_dim
        subset.num_classes = self.num_classes
        return subset

    def split_by_column(self, column: str = 'split') -> Dict[str, 'GroupedMILDataset']:
        """Split dataset by a column."""
        if column not in self.group_df.columns:
            raise ValueError(f"Column '{column}' not in dataset")

        result = {}
        for value in self.group_df[column].unique():
            group_ids = self.group_df[self.group_df[column] == value][self.group_column].tolist()
            result[value] = self.get_subset(group_ids)

        return result

    def random_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Dict[str, 'GroupedMILDataset']:
        """
        Random split by group with optional stratification.

        Ensures no data leakage - all items from same group stay together.
        """
        from sklearn.model_selection import train_test_split

        group_ids = self.group_df[self.group_column].values
        labels = self.group_df.label.values if stratify else None

        test_frac = 1.0 - train_frac - val_frac

        # First split: train+val vs test
        train_val_ids, test_ids = train_test_split(
            group_ids,
            test_size=test_frac,
            stratify=labels if stratify else None,
            random_state=seed,
        )

        # Get labels for train_val subset
        if stratify:
            train_val_mask = self.group_df[self.group_column].isin(train_val_ids)
            train_val_labels = self.group_df[train_val_mask].label.values
        else:
            train_val_labels = None

        # Second split: train vs val
        val_frac_adjusted = val_frac / (train_frac + val_frac)

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_frac_adjusted,
            stratify=train_val_labels,
            random_state=seed,
        )

        return {
            'train': self.get_subset(train_ids.tolist()),
            'val': self.get_subset(val_ids.tolist()),
            'test': self.get_subset(test_ids.tolist()),
        }


# =============================================================================
# Hierarchical Dataset (Preserves Structure) - SKELETON
# =============================================================================

class HierarchicalMILDataset:
    """
    Hierarchical MIL dataset that preserves item structure within groups.

    Instead of concatenating all patches into one bag, this keeps each item
    (e.g., slide) as a separate tensor. This enables two-stage attention:

    1. First stage: Patch attention within each slide -> slide embedding
    2. Second stage: Slide attention across slides -> patient embedding

    Use when:
    - Slides represent distinct samples (different biopsies, regions)
    - You want the model to learn "which patches matter in each slide"
      AND "which slides matter for the patient"

    Created via: MILDataset.group_by('case_id', label_voting='max')

    Example model architecture:
        class HierarchicalMIL(nn.Module):
            def __init__(self):
                self.patch_attention = AttentionMIL(...)  # patches -> slide
                self.slide_attention = AttentionMIL(...)  # slides -> patient

            def forward(self, features_list, slide_mask):
                # features_list: list of [M_i, D] tensors
                slide_embeds = [self.patch_attention(f) for f in features_list]
                slide_embeds = torch.stack(slide_embeds)  # [num_slides, D]
                patient_embed = self.slide_attention(slide_embeds, slide_mask)
                return patient_embed

    TODO: Implement when needed for multi-slide datasets requiring
          hierarchical attention.
    """

    def __init__(self, base_dataset: MILDataset, group_column: str, label_voting: str = 'max'):
        """
        Args:
            base_dataset: The slide-level MILDataset
            group_column: Column to group by (e.g., 'case_id')
            label_voting: How to determine group label ('max', 'maj', 'first')
        """
        self.base = base_dataset
        self.group_column = group_column
        self.label_voting = label_voting
        self.features_dir = base_dataset.features_dir

        # Build grouped dataframe
        self.item_df = base_dataset.df
        self.group_df = self._build_group_df()
        self._index = {row[group_column]: idx for idx, row in self.group_df.iterrows()}

        # Inherit from base
        self.embed_dim = base_dataset.embed_dim
        self.num_classes = self.group_df.label.nunique()

        print(f"\nHierarchical grouping by '{group_column}' (label_voting='{label_voting}'):")
        print(f"  Total items: {len(self.item_df)}")
        print(f"  Total groups: {len(self.group_df)}")
        print(f"  Avg items/group: {len(self.item_df) / len(self.group_df):.2f}")

    def _build_group_df(self) -> pd.DataFrame:
        """Aggregate items into groups."""
        # Determine label aggregation function based on voting method
        if self.label_voting == 'max':
            label_agg = 'max'
        elif self.label_voting == 'maj':
            label_agg = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        elif self.label_voting == 'first':
            label_agg = 'first'
        else:
            raise ValueError(f"Unknown label_voting: {self.label_voting}. "
                           f"Choose from: 'max', 'maj', 'first'")

        grouped = self.item_df.groupby(self.group_column).agg({
            'label': label_agg,
            'slide_id': list,
            'h5_path': list,
            **({'core_id': list} if 'core_id' in self.item_df.columns else {})
        }).reset_index()

        # Check for label conflicts and warn
        conflicts = self._check_label_conflicts()
        if conflicts:
            print(f"  WARNING: {len(conflicts)} groups have inconsistent labels!")
            if len(conflicts) <= 5:
                for group_id, labels in conflicts.items():
                    print(f"    {group_id}: {labels}")
            else:
                print(f"    (showing first 5)")
                for group_id, labels in list(conflicts.items())[:5]:
                    print(f"    {group_id}: {labels}")

        if 'split' in self.item_df.columns:
            split_map = self.item_df.groupby(self.group_column)['split'].first()
            grouped['split'] = grouped[self.group_column].map(split_map)

        grouped['num_items'] = grouped['slide_id'].apply(len)

        return grouped

    def __len__(self) -> int:
        return len(self.group_df)

    def _check_label_conflicts(self) -> Dict[str, List]:
        """Check for groups with inconsistent labels."""
        conflicts = {}
        for group_id, group_df in self.item_df.groupby(self.group_column):
            unique_labels = group_df['label'].unique()
            if len(unique_labels) > 1:
                conflicts[group_id] = unique_labels.tolist()
        return conflicts

    def __iter__(self):
        for _, row in self.group_df.iterrows():
            yield self._load_group(row)

    def __getitem__(self, key: Union[int, str]) -> HierarchicalData:
        if isinstance(key, int):
            row = self.group_df.iloc[key]
        else:
            idx = self._index[key]
            row = self.group_df.iloc[idx]
        return self._load_group(row)

    def _load_group(self, row: pd.Series) -> HierarchicalData:
        """Load all items in the group as separate tensors."""
        features_list = []
        for h5_path in row.h5_path:
            features = load_features(h5_path)
            features_list.append(features)

        return HierarchicalData(
            group_id=row[self.group_column],
            features=features_list,  # List of tensors, NOT concatenated
            label=row.label,
            item_ids=row.slide_id,
            core_ids=row.get('core_id'),
            num_items=row.num_items,
        )

    @property
    def group_ids(self) -> List[str]:
        return self.group_df[self.group_column].tolist()

    @property
    def labels(self) -> List[str]:
        return self.group_df.label.tolist()

    def get_subset(self, group_ids: List[str]) -> 'HierarchicalMILDataset':
        """Create a subset with specific groups."""
        subset = HierarchicalMILDataset.__new__(HierarchicalMILDataset)
        subset.base = self.base
        subset.group_column = self.group_column
        subset.label_voting = self.label_voting
        subset.features_dir = self.features_dir
        subset.item_df = self.item_df[self.item_df[self.group_column].isin(group_ids)].reset_index(drop=True)
        subset.group_df = self.group_df[self.group_df[self.group_column].isin(group_ids)].reset_index(drop=True)
        subset._index = {row[self.group_column]: idx for idx, row in subset.group_df.iterrows()}
        subset.embed_dim = self.embed_dim
        subset.num_classes = self.num_classes
        return subset

    def split_by_column(self, column: str = 'split') -> Dict[str, 'HierarchicalMILDataset']:
        """Split dataset by a column."""
        if column not in self.group_df.columns:
            raise ValueError(f"Column '{column}' not in dataset")

        result = {}
        for value in self.group_df[column].unique():
            group_ids = self.group_df[self.group_df[column] == value][self.group_column].tolist()
            result[value] = self.get_subset(group_ids)

        return result

    def random_split(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        stratify: bool = True,
    ) -> Dict[str, 'HierarchicalMILDataset']:
        """Random split by group with optional stratification."""
        from sklearn.model_selection import train_test_split

        group_ids = self.group_df[self.group_column].values
        labels = self.group_df.label.values if stratify else None

        test_frac = 1.0 - train_frac - val_frac

        train_val_ids, test_ids = train_test_split(
            group_ids,
            test_size=test_frac,
            stratify=labels if stratify else None,
            random_state=seed,
        )

        if stratify:
            train_val_mask = self.group_df[self.group_column].isin(train_val_ids)
            train_val_labels = self.group_df[train_val_mask].label.values
        else:
            train_val_labels = None

        val_frac_adjusted = val_frac / (train_frac + val_frac)

        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_frac_adjusted,
            stratify=train_val_labels,
            random_state=seed,
        )

        return {
            'train': self.get_subset(train_ids.tolist()),
            'val': self.get_subset(val_ids.tolist()),
            'test': self.get_subset(test_ids.tolist()),
        }


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Keep old names working
CaseData = GroupedData
CaseMILDataset = GroupedMILDataset


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import tempfile
    import numpy as np
    import h5py

    print("Testing dataset.py")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock labels CSV
        labels_csv = f"{tmpdir}/labels.csv"
        pd.DataFrame({
            'slide_id': ['slide_001', 'slide_002', 'slide_003', 'slide_004'],
            'case_id': ['case_A', 'case_A', 'case_B', 'case_C'],
            'label': [0, 0, 1, 2],
            'split': ['train', 'train', 'val', 'test'],
        }).to_csv(labels_csv, index=False)

        # Create mock H5 features (only 3 of 4 slides)
        features_dir = f"{tmpdir}/features"
        Path(features_dir).mkdir()
        for slide_id, n_patches in [('slide_001', 100), ('slide_002', 200), ('slide_003', 150)]:
            with h5py.File(f"{features_dir}/{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(n_patches, 1536).astype(np.float32))

        # =================================================================
        # Test 1: MILDataset (slide-level)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 1: MILDataset (slide-level)")
        print("=" * 60)

        dataset = MILDataset(labels_csv, features_dir)

        print(f"\nDataset info:")
        print(f"  Length: {len(dataset)}")
        print(f"  Slide IDs: {dataset.slide_ids}")

        print(f"\nIterating...")
        for slide in dataset:
            print(f"  {slide.slide_id}: {slide.features.shape}, label={slide.label}")

        # =================================================================
        # Test 2: GroupedMILDataset via .concat_by()
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 2: GroupedMILDataset via .concat_by('case_id')")
        print("=" * 60)

        grouped_dataset = dataset.concat_by('case_id')

        print(f"\nGrouped dataset info:")
        print(f"  Length (groups): {len(grouped_dataset)}")
        print(f"  Group IDs: {grouped_dataset.group_ids}")

        print(f"\nIterating over groups...")
        for group in grouped_dataset:
            print(f"  {group.group_id}: {group.features.shape}, "
                  f"label={group.label}, items={group.item_ids}")

        # =================================================================
        # Test 3: HierarchicalMILDataset via .group_by()
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 3: HierarchicalMILDataset via .group_by('case_id')")
        print("=" * 60)

        hier_dataset = dataset.group_by('case_id')

        print(f"\nHierarchical dataset info:")
        print(f"  Length (groups): {len(hier_dataset)}")

        print(f"\nIterating over groups (structure preserved)...")
        for group in hier_dataset:
            shapes = [f.shape for f in group.features]
            print(f"  {group.group_id}: {len(group.features)} items, "
                  f"shapes={shapes}, label={group.label}")

        print(f"\nConverting to padded tensor...")
        group = hier_dataset['case_A']
        padded, mask = group.to_padded_tensor()
        print(f"  Padded shape: {padded.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask: {mask}")

        # =================================================================
        # Test 4: Splitting
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 4: Splitting (no data leakage)")
        print("=" * 60)

        print(f"\nSplit by column (grouped)...")
        splits = grouped_dataset.split_by_column('split')
        for split_name, split_ds in splits.items():
            print(f"  {split_name}: {len(split_ds)} groups")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
