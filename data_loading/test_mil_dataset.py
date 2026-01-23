#!/usr/bin/env python3
"""
Simple tests for MILDataset class.

Tests verify that MILDataset correctly:
1. Initializes with labels CSV and features directory
2. Returns correct dataset length
3. Provides SlideData objects with correct fields
4. Returns features in [M, D] shape (ready for ABMIL)
5. Exposes embed_dim and num_classes
6. Supports random_split() for train/val/test

Run with: python3 test_mil_dataset.py
Or with pytest: pytest test_mil_dataset.py -v
"""

import unittest
import pandas as pd
import numpy as np
import h5py
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from dataset import MILDataset, SlideData


def create_temp_dataset():
    """Create temporary test data with labels CSV and H5 features."""
    tmpdir = TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)

    # Create labels CSV with 4 slides
    labels_csv = tmpdir_path / "labels.csv"
    pd.DataFrame({
        'slide_id': ['slide_001', 'slide_002', 'slide_003', 'slide_004'],
        'case_id': ['case_A', 'case_A', 'case_B', 'case_C'],
        'label': ['0', '0', '1', '2'],  # 3 classes
    }).to_csv(labels_csv, index=False)

    # Create H5 features for 3 of 4 slides (testing join behavior)
    features_dir = tmpdir_path / "features"
    features_dir.mkdir()

    # Different number of patches per slide, embed_dim=1536
    slide_configs = [
        ('slide_001', 100, 1536),
        ('slide_002', 200, 1536),
        ('slide_003', 150, 1536),
    ]

    for slide_id, n_patches, embed_dim in slide_configs:
        with h5py.File(features_dir / f"{slide_id}.h5", 'w') as f:
            features = np.random.randn(n_patches, embed_dim).astype(np.float32)
            f.create_dataset('features', data=features)

    return tmpdir, labels_csv, features_dir


class TestMILDataset(unittest.TestCase):
    """Test suite for MILDataset class."""

    def setUp(self):
        """Create temporary test data before each test."""
        self.tmpdir, self.labels_csv, self.features_dir = create_temp_dataset()

    def tearDown(self):
        """Clean up temporary data after each test."""
        self.tmpdir.cleanup()

    def test_dataset_initialization(self):
        """Test MILDataset initializes correctly."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Should match only slides with both labels AND features (3 out of 4)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.embed_dim, 1536)
        self.assertEqual(dataset.num_classes, 3)

    def test_dataset_length(self):
        """Test len(dataset) returns correct count."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        self.assertEqual(len(dataset), 3)

    def test_getitem_by_index(self):
        """Test dataset[0] returns SlideData with correct fields."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        slide = dataset[0]

        # Check SlideData type and fields
        self.assertIsInstance(slide, SlideData)
        self.assertIsInstance(slide.slide_id, str)
        self.assertIsInstance(slide.features, torch.Tensor)
        self.assertIsInstance(slide.label, str)
        self.assertIsNotNone(slide.case_id)

    def test_features_shape(self):
        """Test slide.features has shape [M, D] ready for ABMIL."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Check each slide has 2D features [M, D]
        for i in range(len(dataset)):
            slide = dataset[i]
            features = slide.features

            # Should be 2D tensor
            self.assertEqual(features.dim(), 2)

            # Shape should be [M, D] where M varies, D is constant
            M, D = features.shape
            self.assertGreater(M, 0)  # At least one patch
            self.assertEqual(D, 1536)  # Embedding dimension

            # Features can be unsqueezed to [1, M, D] for ABMIL
            batched = features.unsqueeze(0)
            self.assertEqual(batched.shape, (1, M, D))

    def test_embed_dim_and_num_classes(self):
        """Test dataset.embed_dim and dataset.num_classes are set correctly."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        self.assertEqual(dataset.embed_dim, 1536)
        self.assertEqual(dataset.num_classes, 3)  # labels: '0', '1', '2'

    def test_random_split(self):
        """Test random_split() returns train/val/test subsets."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        splits = dataset.random_split(train_frac=0.6, val_frac=0.2, seed=42)

        # Should have train, val, test keys
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)

        # All should be MILDataset instances
        self.assertIsInstance(splits['train'], MILDataset)
        self.assertIsInstance(splits['val'], MILDataset)
        self.assertIsInstance(splits['test'], MILDataset)

        # Lengths should sum to original dataset length
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        self.assertEqual(total, len(dataset))

        # Each subset should work correctly
        if len(splits['train']) > 0:
            train_slide = splits['train'][0]
            self.assertIsInstance(train_slide, SlideData)
            self.assertEqual(train_slide.features.dim(), 2)

    def test_iteration(self):
        """Test iterating over dataset yields all slides."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        slides = list(dataset)

        self.assertEqual(len(slides), 3)
        for slide in slides:
            self.assertIsInstance(slide, SlideData)
            self.assertEqual(slide.features.shape[1], 1536)

    def test_getitem_by_slide_id(self):
        """Test accessing slide by slide_id string."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        slide = dataset['slide_002']

        self.assertEqual(slide.slide_id, 'slide_002')
        self.assertEqual(slide.features.shape, (200, 1536))
        self.assertIn(slide.label, ['0', '1', '2'])

    def test_slide_ids_and_labels_properties(self):
        """Test slide_ids and labels properties return correct lists."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        slide_ids = dataset.slide_ids
        labels = dataset.labels

        self.assertEqual(len(slide_ids), 3)
        self.assertEqual(len(labels), 3)
        self.assertTrue(all(isinstance(sid, str) for sid in slide_ids))
        self.assertTrue(all(isinstance(label, str) for label in labels))


if __name__ == '__main__':
    unittest.main(verbosity=2)
