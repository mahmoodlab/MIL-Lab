#!/usr/bin/env python3
"""
Tests for GroupedMILDataset (Early Fusion / "Giant Bag" approach).

Tests verify that concat_by() correctly:
1. Concatenates features from multiple slides into one tensor
2. Total patches = sum of individual slide patches
3. Embedding dimension is preserved
4. Correct slides are grouped by case_id
5. Feature values are preserved (not corrupted)

Run with: python3 test_grouped_dataset.py
Or with pytest: pytest test_grouped_dataset.py -v
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path

from dataset import MILDataset, GroupedMILDataset, GroupedData


class TestGroupedMILDataset(unittest.TestCase):
    """Test suite for GroupedMILDataset (early fusion)."""

    def setUp(self):
        """Create temporary test data with known feature values."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.tmpdir.name) / "features"
        self.features_dir.mkdir()

        # Define slides with KNOWN patch counts and values
        # case_A: 2 slides (slide_001 + slide_002)
        # case_B: 1 slide (slide_003)
        # case_C: 3 slides (slide_004 + slide_005 + slide_006)
        self.slide_configs = {
            'slide_001': {'case_id': 'case_A', 'patches': 100, 'label': '0', 'fill_value': 1.0},
            'slide_002': {'case_id': 'case_A', 'patches': 200, 'label': '0', 'fill_value': 2.0},
            'slide_003': {'case_id': 'case_B', 'patches': 150, 'label': '1', 'fill_value': 3.0},
            'slide_004': {'case_id': 'case_C', 'patches': 50,  'label': '2', 'fill_value': 4.0},
            'slide_005': {'case_id': 'case_C', 'patches': 75,  'label': '2', 'fill_value': 5.0},
            'slide_006': {'case_id': 'case_C', 'patches': 25,  'label': '2', 'fill_value': 6.0},
        }

        self.embed_dim = 1536

        # Create labels CSV
        self.labels_csv = Path(self.tmpdir.name) / "labels.csv"
        rows = []
        for slide_id, config in self.slide_configs.items():
            rows.append({
                'slide_id': slide_id,
                'case_id': config['case_id'],
                'label': config['label'],
            })
            # Create H5 file with known fill value for verification
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                features = np.full(
                    (config['patches'], self.embed_dim),
                    config['fill_value'],
                    dtype=np.float32
                )
                f.create_dataset('features', data=features)

        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)
        self.base_dataset = MILDataset(self.labels_csv, self.features_dir)

    def tearDown(self):
        """Clean up temporary data."""
        self.tmpdir.cleanup()

    def test_concat_by_creates_grouped_dataset(self):
        """Test that concat_by returns a GroupedMILDataset."""
        grouped = self.base_dataset.concat_by('case_id')

        self.assertIsInstance(grouped, GroupedMILDataset)

    def test_grouped_dataset_length(self):
        """Test that grouped dataset has correct number of groups."""
        grouped = self.base_dataset.concat_by('case_id')

        # 3 unique case_ids: case_A, case_B, case_C
        self.assertEqual(len(grouped), 3)

    def test_features_are_concatenated_not_separate(self):
        """Test that features are concatenated into ONE tensor (Giant Bag)."""
        grouped = self.base_dataset.concat_by('case_id')

        case_a = grouped['case_A']

        # Should be a single tensor, NOT a list
        self.assertIsInstance(case_a.features, torch.Tensor)
        self.assertEqual(case_a.features.dim(), 2)  # [M_total, D]

    def test_total_patches_equals_sum_of_slides(self):
        """Test that concatenated patches = sum of individual slide patches."""
        grouped = self.base_dataset.concat_by('case_id')

        # case_A: slide_001 (100) + slide_002 (200) = 300 patches
        case_a = grouped['case_A']
        self.assertEqual(case_a.features.shape[0], 100 + 200)

        # case_B: slide_003 (150) = 150 patches
        case_b = grouped['case_B']
        self.assertEqual(case_b.features.shape[0], 150)

        # case_C: slide_004 (50) + slide_005 (75) + slide_006 (25) = 150 patches
        case_c = grouped['case_C']
        self.assertEqual(case_c.features.shape[0], 50 + 75 + 25)

    def test_embedding_dimension_preserved(self):
        """Test that embedding dimension is preserved after concatenation."""
        grouped = self.base_dataset.concat_by('case_id')

        for group in grouped:
            self.assertEqual(group.features.shape[1], self.embed_dim)

    def test_feature_values_preserved(self):
        """Test that feature values are not corrupted during concatenation."""
        grouped = self.base_dataset.concat_by('case_id')

        # case_A has slide_001 (fill=1.0, 100 patches) + slide_002 (fill=2.0, 200 patches)
        case_a = grouped['case_A']
        features = case_a.features

        # First 100 patches should have value 1.0 (from slide_001)
        self.assertTrue(torch.allclose(
            features[:100, :],
            torch.full((100, self.embed_dim), 1.0)
        ))

        # Next 200 patches should have value 2.0 (from slide_002)
        self.assertTrue(torch.allclose(
            features[100:300, :],
            torch.full((200, self.embed_dim), 2.0)
        ))

    def test_item_ids_track_source_slides(self):
        """Test that item_ids correctly tracks which slides were concatenated."""
        grouped = self.base_dataset.concat_by('case_id')

        case_a = grouped['case_A']
        self.assertEqual(set(case_a.item_ids), {'slide_001', 'slide_002'})

        case_c = grouped['case_C']
        self.assertEqual(set(case_c.item_ids), {'slide_004', 'slide_005', 'slide_006'})

    def test_num_items_correct(self):
        """Test that num_items reflects number of slides in group."""
        grouped = self.base_dataset.concat_by('case_id')

        self.assertEqual(grouped['case_A'].num_items, 2)
        self.assertEqual(grouped['case_B'].num_items, 1)
        self.assertEqual(grouped['case_C'].num_items, 3)

    def test_getitem_returns_grouped_data(self):
        """Test that __getitem__ returns GroupedData objects."""
        grouped = self.base_dataset.concat_by('case_id')

        # Access by index
        group = grouped[0]
        self.assertIsInstance(group, GroupedData)

        # Access by group_id
        group = grouped['case_A']
        self.assertIsInstance(group, GroupedData)
        self.assertEqual(group.group_id, 'case_A')

    def test_iteration_yields_all_groups(self):
        """Test that iterating yields all groups."""
        grouped = self.base_dataset.concat_by('case_id')

        groups = list(grouped)
        self.assertEqual(len(groups), 3)

        group_ids = {g.group_id for g in groups}
        self.assertEqual(group_ids, {'case_A', 'case_B', 'case_C'})

    def test_embed_dim_inherited(self):
        """Test that embed_dim is accessible on grouped dataset."""
        grouped = self.base_dataset.concat_by('case_id')

        self.assertEqual(grouped.embed_dim, self.embed_dim)

    def test_single_slide_case_unchanged(self):
        """Test that cases with single slide have unchanged features."""
        grouped = self.base_dataset.concat_by('case_id')

        # case_B has only slide_003
        case_b = grouped['case_B']

        # Shape should match original slide
        self.assertEqual(case_b.features.shape, (150, self.embed_dim))

        # Values should be preserved (slide_003 has fill_value=3.0)
        self.assertTrue(torch.allclose(
            case_b.features,
            torch.full((150, self.embed_dim), 3.0)
        ))

    def test_random_split_no_case_leakage(self):
        """Test that random_split keeps all slides from same case together."""
        grouped = self.base_dataset.concat_by('case_id')

        splits = grouped.random_split(train_frac=0.5, val_frac=0.25, seed=42, stratify=False)

        train_cases = set(splits['train'].group_ids)
        val_cases = set(splits['val'].group_ids)
        test_cases = set(splits['test'].group_ids)

        # No overlap between splits
        self.assertTrue(train_cases.isdisjoint(val_cases))
        self.assertTrue(train_cases.isdisjoint(test_cases))
        self.assertTrue(val_cases.isdisjoint(test_cases))

        # All cases accounted for
        all_cases = train_cases | val_cases | test_cases
        self.assertEqual(all_cases, {'case_A', 'case_B', 'case_C'})


class TestEarlyFusionEndToEnd(unittest.TestCase):
    """End-to-end test simulating real training scenario."""

    def setUp(self):
        """Create test data."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.tmpdir.name) / "features"
        self.features_dir.mkdir()

        # Patient with 2 slides
        slides = [
            ('slide_A1', 'patient_A', 100, '1'),
            ('slide_A2', 'patient_A', 150, '1'),
        ]

        rows = []
        for slide_id, case_id, patches, label in slides:
            rows.append({'slide_id': slide_id, 'case_id': case_id, 'label': label})
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(patches, 512).astype(np.float32))

        self.labels_csv = Path(self.tmpdir.name) / "labels.csv"
        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_giant_bag_ready_for_mil_model(self):
        """Test that Giant Bag output is ready for MIL model input."""
        from pytorch_adapter import MILDatasetAdapter

        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id')
        adapter = MILDatasetAdapter(grouped)

        # Get a sample
        features, label = adapter[0]

        # Features should be [M_total, D] - ready for MIL
        self.assertEqual(features.dim(), 2)
        self.assertEqual(features.shape[0], 100 + 150)  # Concatenated patches
        self.assertEqual(features.shape[1], 512)

        # Can be batched to [1, M, D] for model
        batched = features.unsqueeze(0)
        self.assertEqual(batched.shape, (1, 250, 512))


if __name__ == '__main__':
    unittest.main(verbosity=2)
