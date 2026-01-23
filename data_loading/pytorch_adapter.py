#!/usr/bin/env python3
"""
PyTorch adapter for MILDataset.

Bridges the MILDataset to PyTorch DataLoader with support for:
- Variable-length bags (padding/masking)
- Weighted sampling for class imbalance
- Label encoding
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .dataset import MILDataset


class MILDatasetAdapter(Dataset):
    """
    Wraps MILDataset as a PyTorch Dataset.

    Converts string labels to integer indices and returns
    (features, label) tuples ready for training.
    """

    def __init__(
        self,
        mil_dataset: 'MILDataset',
        label_map: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            mil_dataset: MILDataset instance
            label_map: Optional mapping from string labels to integers.
                       If None, creates one from unique labels.
        """
        self.mil_dataset = mil_dataset

        # Create label map if not provided
        if label_map is None:
            unique_labels = sorted(set(mil_dataset.labels))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self) -> int:
        return len(self.mil_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        slide = self.mil_dataset[idx]
        features = slide.features  # [M, D]
        label = torch.tensor(self.label_map[slide.label], dtype=torch.long)
        return features, label

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def embed_dim(self) -> int:
        return self.mil_dataset.embed_dim

    def get_labels(self) -> List[int]:
        """Get integer labels for all samples (useful for weighted sampling)."""
        return [self.label_map[label] for label in self.mil_dataset.labels]


class HierarchicalMILDatasetAdapter(Dataset):
    """
    Wraps HierarchicalMILDataset as a PyTorch Dataset.
    Returns (features_list, label) where features_list is a List[torch.Tensor].
    """

    def __init__(
        self,
        hier_dataset: 'HierarchicalMILDataset',
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.hier_dataset = hier_dataset

        if label_map is None:
            unique_labels = sorted(set(hier_dataset.labels))
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    def __len__(self) -> int:
        return len(self.hier_dataset)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        hier_data = self.hier_dataset[idx]
        features_list = hier_data.features  # List of [M_i, D]
        label = torch.tensor(self.label_map[hier_data.label], dtype=torch.long)
        return features_list, label

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def embed_dim(self) -> int:
        return self.hier_dataset.embed_dim

    def get_labels(self) -> List[int]:
        return [self.label_map[label] for label in self.hier_dataset.labels]


def hierarchical_collate_fn(
    batch: List[Tuple[List[torch.Tensor], torch.Tensor]]
) -> Tuple[List[List[torch.Tensor]], torch.Tensor, List]:
    """
    Collate function for hierarchical MIL.
    Each item in batch is (List[torch.Tensor], label).
    Returns (padded_features_list, labels, masks_list).
    """
    features_lists, labels = zip(*batch)
    labels = torch.stack(labels)
    
    return list(features_lists), labels, []


def mil_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable-length bags.

    Pads features to max length and creates attention masks.

    Args:
        batch: List of (features, label) tuples
               features: [M_i, D] tensor (variable num patches)
               label: scalar tensor

    Returns:
        padded_features: [B, M_max, D] tensor
        labels: [B] tensor
        mask: [B, M_max] attention mask (1=valid, 0=padding)
    """
    features_list, labels = zip(*batch)

    batch_size = len(features_list)
    max_patches = max(f.shape[0] for f in features_list)
    embed_dim = features_list[0].shape[1]

    # Pad features
    padded = torch.zeros(batch_size, max_patches, embed_dim)
    mask = torch.zeros(batch_size, max_patches)

    for i, features in enumerate(features_list):
        num_patches = features.shape[0]
        padded[i, :num_patches, :] = features
        mask[i, :num_patches] = 1.0

    labels = torch.stack(labels)

    return padded, labels, mask


def create_dataloader(
    mil_dataset: Union['MILDataset', 'HierarchicalMILDataset'],
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    weighted_sampling: bool = False,
    label_map: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Tuple[DataLoader, Dataset]:
    """
    Create a DataLoader from a MILDataset or HierarchicalMILDataset.
    """
    from .dataset import HierarchicalMILDataset, GroupedMILDataset, MILDataset
    
    if isinstance(mil_dataset, HierarchicalMILDataset):
        adapter = HierarchicalMILDatasetAdapter(mil_dataset, label_map=label_map)
        collate_fn = hierarchical_collate_fn
    else:
        adapter = MILDatasetAdapter(mil_dataset, label_map=label_map)
        collate_fn = mil_collate_fn if batch_size > 1 else None

    sampler = None
    if weighted_sampling:
        labels = adapter.get_labels()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]

        generator = torch.Generator()
        generator.manual_seed(seed)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False  # Cannot use both sampler and shuffle

    # Only override collate_fn if it's not already set (e.g., for hierarchical)
    if not isinstance(mil_dataset, HierarchicalMILDataset):
        collate_fn = mil_collate_fn if batch_size > 1 else None

    loader = DataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return loader, adapter
