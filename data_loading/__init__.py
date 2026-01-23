"""
Data loading utilities for MIL-Lab.

Supports multiple levels of aggregation:
    1. Slide-level: MILDataset (one bag per H5 file)
    2. Grouped: GroupedMILDataset via .concat_by() (flat concatenation)
    3. Hierarchical: HierarchicalMILDataset via .group_by() (preserves structure)

Usage:
    from data_loading import MILDataset, create_dataloader

    # Slide-level (e.g., PANDA: 1 patient = 1 slide)
    dataset = MILDataset('labels.csv', 'features/')

    # Patient-level with concatenation (multi-slide patients)
    patient_dataset = dataset.concat_by('case_id')

    # Patient-level with hierarchy (for two-stage attention)
    patient_dataset = dataset.group_by('case_id')

    # Create PyTorch DataLoader
    loader, adapter = create_dataloader(dataset)
"""

from .feature_loader import (
    load_features,
    load_features_batch,
    get_slide_ids,
    get_embed_dim,
    CLAMFeatureLoader,
    prepare_for_mil,
    batch_for_mil,
)

from .dataset import (
    # Functions
    load_labels,
    get_available_features,
    join_labels_and_features,
    # Data containers
    SlideData,
    GroupedData,
    HierarchicalData,
    # Dataset classes
    MILDataset,
    GroupedMILDataset,
    HierarchicalMILDataset,
    # Backwards compatibility aliases
    CaseData,
    CaseMILDataset,
)

from .pytorch_adapter import (
    MILDatasetAdapter,
    mil_collate_fn,
    create_dataloader,
)

__all__ = [
    # Features (H5)
    'load_features',
    'load_features_batch',
    'get_slide_ids',
    'get_embed_dim',
    'CLAMFeatureLoader',
    'prepare_for_mil',
    'batch_for_mil',
    # Labels (CSV)
    'load_labels',
    'get_available_features',
    'join_labels_and_features',
    # Data containers
    'SlideData',
    'GroupedData',
    'HierarchicalData',
    # Slide-level dataset
    'MILDataset',
    # Grouped dataset (flat concatenation)
    'GroupedMILDataset',
    # Hierarchical dataset (preserves structure)
    'HierarchicalMILDataset',
    # Backwards compatibility
    'CaseData',       # alias for GroupedData
    'CaseMILDataset', # alias for GroupedMILDataset
    # PyTorch adapter
    'MILDatasetAdapter',
    'mil_collate_fn',
    'create_dataloader',
]
