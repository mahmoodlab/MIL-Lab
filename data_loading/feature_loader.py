#!/usr/bin/env python3
"""
CLAM Feature Loader

Simple, single-responsibility module for loading CLAM-extracted HDF5 features
and preparing them for any MIL model.

Design:
    - Separation of concerns: data loading is independent of model choice
    - Dagster-ready: can be swapped as an asset in a data pipeline
    - No labels: downstream tasks handle labeling/splits
    - Returns [M, D] tensors ready to be batched to [B, M, D] for MIL models

Usage:
    from clam_dataloader import load_features, load_features_batch, CLAMFeatureLoader

    # Single slide
    features = load_features('/path/to/slide.h5')  # -> [M, D] tensor

    # Multiple slides
    features_dict = load_features_batch('/path/to/h5_dir', ['slide_001', 'slide_002'])

    # Iterator for large datasets
    loader = CLAMFeatureLoader('/path/to/h5_dir')
    for slide_id, features in loader:
        # process features...
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Tuple, Union


def load_features(h5_path: Union[str, Path]) -> torch.Tensor:
    """
    Load features from a single CLAM HDF5 file.

    Args:
        h5_path: Path to .h5 file

    Returns:
        features: [M, D] tensor where M=num_patches, D=embed_dim
    """
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float()

    # Normalize shape: handle both 2D (M, D) and 3D (1, M, D) formats
    if features.dim() == 3 and features.shape[0] == 1:
        features = features.squeeze(0)

    # Handle single-patch edge case: (D,) -> (1, D)
    if features.dim() == 1:
        features = features.unsqueeze(0)

    return features


def load_features_batch(
    features_dir: Union[str, Path],
    slide_ids: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Load features for multiple slides.

    Args:
        features_dir: Directory containing .h5 files
        slide_ids: List of slide IDs (without .h5 extension)

    Returns:
        Dictionary mapping slide_id -> [M, D] tensor
    """
    features_dir = Path(features_dir)
    result = {}

    for slide_id in slide_ids:
        h5_path = features_dir / f"{slide_id}.h5"
        if h5_path.exists():
            result[slide_id] = load_features(h5_path)

    return result


def get_slide_ids(features_dir: Union[str, Path]) -> List[str]:
    """
    Get all slide IDs from a features directory.

    Args:
        features_dir: Directory containing .h5 files

    Returns:
        List of slide IDs (filenames without .h5 extension)
    """
    features_dir = Path(features_dir)
    return sorted([p.stem for p in features_dir.glob('*.h5')])


def get_embed_dim(features_dir: Union[str, Path]) -> int:
    """
    Infer embedding dimension from the first .h5 file.

    Args:
        features_dir: Directory containing .h5 files

    Returns:
        Embedding dimension (e.g., 1024, 1536)
    """
    features_dir = Path(features_dir)
    h5_files = list(features_dir.glob('*.h5'))

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files in {features_dir}")

    with h5py.File(h5_files[0], 'r') as f:
        return f['features'].shape[-1]  # Last dim is always embed_dim


class CLAMFeatureLoader:
    """
    Iterator for loading CLAM features from a directory.

    Yields (slide_id, features) tuples without loading all into memory.

    Example:
        loader = CLAMFeatureLoader('/path/to/features')
        for slide_id, features in loader:
            # features is [M, D] tensor
            pass
    """

    def __init__(
        self,
        features_dir: Union[str, Path],
        slide_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            features_dir: Directory containing .h5 files
            slide_ids: Optional subset of slides to load (default: all)
        """
        self.features_dir = Path(features_dir)

        if slide_ids is None:
            self.slide_ids = get_slide_ids(features_dir)
        else:
            self.slide_ids = list(slide_ids)

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for slide_id in self.slide_ids:
            h5_path = self.features_dir / f"{slide_id}.h5"
            if h5_path.exists():
                yield slide_id, load_features(h5_path)

    def __getitem__(self, slide_id: str) -> torch.Tensor:
        """Load features for a specific slide."""
        h5_path = self.features_dir / f"{slide_id}.h5"
        return load_features(h5_path)

    @property
    def embed_dim(self) -> int:
        """Get embedding dimension."""
        return get_embed_dim(self.features_dir)


# =============================================================================
# Convenience functions for pipeline integration
# =============================================================================

def prepare_for_mil(features: torch.Tensor) -> torch.Tensor:
    """
    Prepare features tensor for MIL model input.

    Ensures tensor is [B, M, D] format expected by MIL models.

    Args:
        features: [M, D] tensor from load_features()

    Returns:
        [1, M, D] tensor ready for MIL forward pass
    """
    if features.dim() == 2:
        return features.unsqueeze(0)  # [M, D] -> [1, M, D]
    return features


def batch_for_mil(features_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch multiple slides for MIL model input with padding.

    Args:
        features_list: List of [M_i, D] tensors (variable num patches)

    Returns:
        padded_features: [B, M_max, D] tensor
        mask: [B, M_max] attention mask (1=valid, 0=padding)
    """
    batch_size = len(features_list)
    max_patches = max(f.shape[0] for f in features_list)
    embed_dim = features_list[0].shape[1]

    padded = torch.zeros(batch_size, max_patches, embed_dim)
    mask = torch.zeros(batch_size, max_patches)

    for i, features in enumerate(features_list):
        num_patches = features.shape[0]
        padded[i, :num_patches, :] = features
        mask[i, :num_patches] = 1.0

    return padded, mask


