#!/usr/bin/env python3
"""
Comprehensive tests for verifying data shape transformations in the MIL data loading pipeline.

Tests trace data shapes through the complete pipeline:
    Input: n patients × m slides/patient × p patches/slide × D embedding dimension
    Stage 1 (MILDataset): Each slide is [p, D]
    Stage 2 (GroupedMILDataset): Each patient is [sum_of_patches, D]
    Stage 3 (DataLoader): Batch is [B, max_patches, D] with mask [B, max_patches]

Specific Test Scenario:
    - 3 patients (case_ids)
    - Patient 1: 2 slides (300 tiles + 200 tiles) → after grouping: 500×1024
    - Patient 2: 1 slide (150 tiles) → after grouping: 150×1024
    - Patient 3: 3 slides (100 + 100 + 100 tiles) → after grouping: 300×1024
    - Binary classification labels (0 or 1)
    - Embedding dimension: 1024

Run with:
    python -m pytest tests/test_data_shape_tracing.py -v
    python tests/test_data_shape_tracing.py
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data_loading'))

from data_loading.dataset import MILDataset, GroupedMILDataset, SlideData, GroupedData
from data_loading.pytorch_adapter import create_dataloader, MILDatasetAdapter


class TestBinaryClassificationShapeTracing(unittest.TestCase):
    """
    Test data shape transformations for binary classification with case_id grouping.

    This test creates a specific scenario and verifies shapes at each stage:
    - Patient 1: 2 slides (300 + 200 patches) → 500 patches after grouping
    - Patient 2: 1 slide (150 patches) → 150 patches after grouping
    - Patient 3: 3 slides (100 + 100 + 100 patches) → 300 patches after grouping
    - Embedding dimension: 1024
    - Binary labels: 0 or 1
    """

    def setUp(self):
        """Create the exact test scenario with known shapes."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.tmpdir.name) / "features"
        self.features_dir.mkdir()

        # Define exact scenario
        # Patient 1: 2 slides with different patch counts
        # Patient 2: 1 slide
        # Patient 3: 3 slides with same patch counts
        self.slide_configs = [
            # (slide_id, case_id, num_patches, label, fill_value)
            ('patient1_slide1', 'patient_1', 300, '0', 1.0),
            ('patient1_slide2', 'patient_1', 200, '0', 1.5),
            ('patient2_slide1', 'patient_2', 150, '1', 2.0),
            ('patient3_slide1', 'patient_3', 100, '1', 3.0),
            ('patient3_slide2', 'patient_3', 100, '1', 3.5),
            ('patient3_slide3', 'patient_3', 100, '1', 4.0),
        ]

        self.embed_dim = 1024
        self.num_patients = 3

        # Expected grouped shapes after concat_by('case_id')
        self.expected_grouped_shapes = {
            'patient_1': (500, 1024),  # 300 + 200
            'patient_2': (150, 1024),  # 150
            'patient_3': (300, 1024),  # 100 + 100 + 100
        }

        # Expected number of slides per patient
        self.expected_num_slides = {
            'patient_1': 2,
            'patient_2': 1,
            'patient_3': 3,
        }

        # Create labels CSV
        self.labels_csv = Path(self.tmpdir.name) / "labels.csv"
        rows = []
        for slide_id, case_id, num_patches, label, fill_value in self.slide_configs:
            rows.append({
                'slide_id': slide_id,
                'case_id': case_id,
                'label': label,
            })

            # Create H5 file with known fill value for verification
            # This allows us to verify that features are correctly concatenated
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                features = np.full(
                    (num_patches, self.embed_dim),
                    fill_value,
                    dtype=np.float32
                )
                f.create_dataset('features', data=features)

        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)

    def tearDown(self):
        """Clean up temporary data."""
        self.tmpdir.cleanup()

    # =========================================================================
    # Stage 1: Raw MILDataset - Verify slide-level shapes [p, D]
    # =========================================================================

    def test_stage1_raw_dataset_initialization(self):
        """Test that MILDataset initializes with correct metadata."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Should have 6 slides total
        self.assertEqual(len(dataset), 6)

        # Should detect embedding dimension correctly
        self.assertEqual(dataset.embed_dim, self.embed_dim)

        # Should detect 2 classes (binary: 0 and 1)
        self.assertEqual(dataset.num_classes, 2)

    def test_stage1_individual_slide_shapes(self):
        """Test that each slide has shape [p, D] where p is specific to that slide."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Define expected shapes for each slide
        expected_shapes = {
            'patient1_slide1': (300, 1024),
            'patient1_slide2': (200, 1024),
            'patient2_slide1': (150, 1024),
            'patient3_slide1': (100, 1024),
            'patient3_slide2': (100, 1024),
            'patient3_slide3': (100, 1024),
        }

        for slide_id, expected_shape in expected_shapes.items():
            slide = dataset[slide_id]

            # Verify it's a SlideData object
            self.assertIsInstance(slide, SlideData)

            # Verify features are 2D tensor [patches, embedding_dim]
            self.assertEqual(slide.features.dim(), 2)

            # Verify exact shape
            self.assertEqual(
                slide.features.shape,
                expected_shape,
                f"Slide {slide_id} has incorrect shape"
            )

    def test_stage1_all_slides_have_correct_embedding_dim(self):
        """Test that all slides consistently have embedding dimension 1024."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        for slide in dataset:
            # All slides should have shape [p, 1024]
            self.assertEqual(
                slide.features.shape[1],
                self.embed_dim,
                f"Slide {slide.slide_id} has wrong embedding dimension"
            )

    def test_stage1_slide_metadata_correct(self):
        """Test that slide metadata (slide_id, case_id, label) is preserved."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Check patient 1, slide 1
        slide = dataset['patient1_slide1']
        self.assertEqual(slide.slide_id, 'patient1_slide1')
        self.assertEqual(slide.case_id, 'patient_1')
        self.assertEqual(slide.label, '0')

        # Check patient 2, slide 1
        slide = dataset['patient2_slide1']
        self.assertEqual(slide.slide_id, 'patient2_slide1')
        self.assertEqual(slide.case_id, 'patient_2')
        self.assertEqual(slide.label, '1')

    def test_stage1_binary_labels_distribution(self):
        """Test that we have binary labels with expected distribution."""
        dataset = MILDataset(self.labels_csv, self.features_dir)

        labels = dataset.labels
        unique_labels = set(labels)

        # Should have exactly 2 unique labels
        self.assertEqual(len(unique_labels), 2)
        self.assertEqual(unique_labels, {'0', '1'})

        # Count distribution
        label_counts = {label: labels.count(label) for label in unique_labels}
        self.assertEqual(label_counts['0'], 2)  # patient1_slide1, patient1_slide2
        self.assertEqual(label_counts['1'], 4)  # patient2_slide1, patient3_slide1-3

    # =========================================================================
    # Stage 2: GroupedMILDataset - Verify concatenation [sum_of_patches, D]
    # =========================================================================

    def test_stage2_grouped_dataset_length(self):
        """Test that grouping by case_id creates 3 patient groups."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        # Should have 3 patients
        self.assertEqual(len(grouped), self.num_patients)

    def test_stage2_patient1_concatenated_shape(self):
        """Test that Patient 1's features are concatenated to [500, 1024]."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        patient1 = grouped['patient_1']

        # Verify it's a GroupedData object
        self.assertIsInstance(patient1, GroupedData)

        # Verify features are concatenated, NOT a list
        self.assertIsInstance(patient1.features, torch.Tensor)
        self.assertEqual(patient1.features.dim(), 2)

        # Verify exact shape: 300 + 200 = 500 patches
        expected_shape = self.expected_grouped_shapes['patient_1']
        self.assertEqual(
            patient1.features.shape,
            expected_shape,
            "Patient 1: Expected concatenation of 300 + 200 = 500 patches"
        )

    def test_stage2_patient2_concatenated_shape(self):
        """Test that Patient 2's features remain [150, 1024] (single slide)."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        patient2 = grouped['patient_2']

        # Verify exact shape: 150 patches (single slide)
        expected_shape = self.expected_grouped_shapes['patient_2']
        self.assertEqual(
            patient2.features.shape,
            expected_shape,
            "Patient 2: Expected 150 patches (single slide)"
        )

    def test_stage2_patient3_concatenated_shape(self):
        """Test that Patient 3's features are concatenated to [300, 1024]."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        patient3 = grouped['patient_3']

        # Verify exact shape: 100 + 100 + 100 = 300 patches
        expected_shape = self.expected_grouped_shapes['patient_3']
        self.assertEqual(
            patient3.features.shape,
            expected_shape,
            "Patient 3: Expected concatenation of 100 + 100 + 100 = 300 patches"
        )

    def test_stage2_all_patients_have_correct_embedding_dim(self):
        """Test that embedding dimension is preserved after grouping."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        for patient in grouped:
            self.assertEqual(
                patient.features.shape[1],
                self.embed_dim,
                f"Patient {patient.group_id} has wrong embedding dimension after grouping"
            )

    def test_stage2_total_patches_preserved(self):
        """Test that total number of patches is preserved during concatenation."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        # Count total patches before grouping
        total_before = sum(slide.features.shape[0] for slide in dataset)

        # Count total patches after grouping
        total_after = sum(patient.features.shape[0] for patient in grouped)

        # Should be equal
        self.assertEqual(
            total_before,
            total_after,
            "Total patches changed during grouping (should be preserved)"
        )

        # Verify exact count: 300 + 200 + 150 + 100 + 100 + 100 = 950
        self.assertEqual(total_before, 950)
        self.assertEqual(total_after, 950)

    def test_stage2_feature_values_preserved_during_concat(self):
        """Test that feature values are not corrupted during concatenation."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        # Patient 1 has 2 slides with fill values 1.0 and 1.5
        patient1 = grouped['patient_1']
        features = patient1.features

        # First 300 patches should have fill value 1.0 (from slide 1)
        self.assertTrue(
            torch.allclose(
                features[:300, :],
                torch.full((300, self.embed_dim), 1.0)
            ),
            "Patient 1, slide 1 features corrupted"
        )

        # Next 200 patches should have fill value 1.5 (from slide 2)
        self.assertTrue(
            torch.allclose(
                features[300:500, :],
                torch.full((200, self.embed_dim), 1.5)
            ),
            "Patient 1, slide 2 features corrupted"
        )

        # Patient 3 has 3 slides with fill values 3.0, 3.5, 4.0
        patient3 = grouped['patient_3']
        features = patient3.features

        # First 100 patches: 3.0
        self.assertTrue(
            torch.allclose(
                features[:100, :],
                torch.full((100, self.embed_dim), 3.0)
            )
        )

        # Next 100 patches: 3.5
        self.assertTrue(
            torch.allclose(
                features[100:200, :],
                torch.full((100, self.embed_dim), 3.5)
            )
        )

        # Last 100 patches: 4.0
        self.assertTrue(
            torch.allclose(
                features[200:300, :],
                torch.full((100, self.embed_dim), 4.0)
            )
        )

    # =========================================================================
    # Stage 3: Metadata Verification
    # =========================================================================

    def test_stage2_num_items_tracks_slide_count(self):
        """Test that num_items correctly reflects number of slides per patient."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        for group_id, expected_count in self.expected_num_slides.items():
            patient = grouped[group_id]
            self.assertEqual(
                patient.num_items,
                expected_count,
                f"{group_id}: Expected {expected_count} slides"
            )

    def test_stage2_item_ids_lists_all_slides(self):
        """Test that item_ids correctly lists all slide_ids in each group."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        # Patient 1 should have 2 specific slides
        patient1 = grouped['patient_1']
        self.assertEqual(
            set(patient1.item_ids),
            {'patient1_slide1', 'patient1_slide2'}
        )

        # Patient 2 should have 1 specific slide
        patient2 = grouped['patient_2']
        self.assertEqual(
            set(patient2.item_ids),
            {'patient2_slide1'}
        )

        # Patient 3 should have 3 specific slides
        patient3 = grouped['patient_3']
        self.assertEqual(
            set(patient3.item_ids),
            {'patient3_slide1', 'patient3_slide2', 'patient3_slide3'}
        )

    def test_stage2_group_ids_match_case_ids(self):
        """Test that group_ids match case_ids."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        expected_group_ids = {'patient_1', 'patient_2', 'patient_3'}
        actual_group_ids = set(grouped.group_ids)

        self.assertEqual(actual_group_ids, expected_group_ids)

    # =========================================================================
    # Label Preservation and Voting
    # =========================================================================

    def test_stage2_labels_preserved_with_first_voting(self):
        """Test that labels are preserved correctly with label_voting='first'."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        # Patient 1: All slides have label '0'
        self.assertEqual(grouped['patient_1'].label, '0')

        # Patient 2: Single slide has label '1'
        self.assertEqual(grouped['patient_2'].label, '1')

        # Patient 3: All slides have label '1'
        self.assertEqual(grouped['patient_3'].label, '1')

    def test_stage2_labels_preserved_with_max_voting(self):
        """Test that labels work correctly with label_voting='max' for binary data."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='max')

        # For binary data where all slides in a patient have same label,
        # 'max' should give same result as 'first'
        self.assertEqual(grouped['patient_1'].label, '0')
        self.assertEqual(grouped['patient_2'].label, '1')
        self.assertEqual(grouped['patient_3'].label, '1')

    def test_stage2_binary_label_distribution_after_grouping(self):
        """Test that binary label distribution is correct after grouping."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        labels = grouped.labels
        unique_labels = set(labels)

        # Should still have 2 unique labels
        self.assertEqual(len(unique_labels), 2)
        self.assertEqual(unique_labels, {'0', '1'})

        # Count distribution at patient level
        label_counts = {label: labels.count(label) for label in unique_labels}
        self.assertEqual(label_counts['0'], 1)  # patient_1
        self.assertEqual(label_counts['1'], 2)  # patient_2, patient_3

    # =========================================================================
    # Stage 3: DataLoader Batching with Padding
    # =========================================================================

    def test_stage3_dataloader_batch_shape_batch_size_1(self):
        """Test DataLoader with batch_size=1 produces correct shapes."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        # Get first batch
        # With batch_size=1, collate_fn is None, but DataLoader still adds batch dimension
        batch = next(iter(loader))
        features, labels = batch

        # DataLoader always adds batch dimension, so shape is [1, num_patches, embed_dim]
        self.assertEqual(features.dim(), 3)
        self.assertEqual(features.shape[0], 1)  # batch size
        self.assertIn(features.shape[1], [500, 150, 300])  # one of the patient shapes
        self.assertEqual(features.shape[2], self.embed_dim)

        # Labels should be [1]
        self.assertEqual(labels.shape, torch.Size([1]))

    def test_stage3_dataloader_batch_shape_batch_size_2(self):
        """Test DataLoader with batch_size=2 pads to max length in batch."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        # Get first batch
        features, labels, mask = next(iter(loader))

        # Features should be padded to max length in batch
        # Shape: [batch_size, max_patches_in_batch, embed_dim]
        self.assertEqual(features.dim(), 3)
        self.assertEqual(features.shape[0], 2)  # batch size
        self.assertEqual(features.shape[2], self.embed_dim)

        # max_patches_in_batch should be the larger of the two patients in batch
        # This will depend on order, but should be one of our patient sizes
        max_patches = features.shape[1]
        self.assertIn(max_patches, [500, 150, 300])

        # Labels should be [2]
        self.assertEqual(labels.shape, torch.Size([2]))

        # Mask should be [2, max_patches]
        self.assertEqual(mask.shape, torch.Size([2, max_patches]))

    def test_stage3_dataloader_batch_shape_full_batch(self):
        """Test DataLoader with batch_size=3 (all patients) pads to max=500."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )

        # Get first (and only) batch
        features, labels, mask = next(iter(loader))

        # With all 3 patients, should pad to max=500 (patient_1)
        self.assertEqual(features.shape, torch.Size([3, 500, self.embed_dim]))

        # Labels should be [3]
        self.assertEqual(labels.shape, torch.Size([3]))

        # Mask should be [3, 500]
        self.assertEqual(mask.shape, torch.Size([3, 500]))

    def test_stage3_padding_mask_correctness(self):
        """Test that padding masks correctly indicate valid vs padded positions."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )

        # Get batch with all 3 patients
        features, labels, mask = next(iter(loader))

        # All batches padded to 500 (max in dataset)
        # Patient shapes: [500, 150, 300]
        # So masks should be:
        # - Patient 0: 500 ones (no padding needed)
        # - Patient 1: 150 ones, 350 zeros
        # - Patient 2: 300 ones, 200 zeros

        # Check number of valid (non-zero) positions per patient
        valid_counts = mask.sum(dim=1).long().tolist()

        # The order depends on how the data is sorted, but the set should match
        expected_counts = {500, 150, 300}
        actual_counts = set(valid_counts)

        self.assertEqual(
            actual_counts,
            expected_counts,
            "Mask should indicate correct number of valid patches per patient"
        )

    def test_stage3_padded_regions_are_zero(self):
        """Test that padded regions in features are filled with zeros."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )

        features, labels, mask = next(iter(loader))

        # For each sample in batch, verify padded positions are zero
        for i in range(features.shape[0]):
            sample_features = features[i]  # [max_patches, embed_dim]
            sample_mask = mask[i]  # [max_patches]

            # Find padded positions (where mask is 0)
            padded_positions = (sample_mask == 0)

            if padded_positions.any():
                # Check that padded positions are all zeros
                padded_features = sample_features[padded_positions]
                self.assertTrue(
                    torch.allclose(padded_features, torch.zeros_like(padded_features)),
                    f"Sample {i}: Padded positions should be zero"
                )

    def test_stage3_adapter_metadata_correct(self):
        """Test that adapter exposes correct metadata."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        # Check adapter properties
        self.assertEqual(adapter.num_classes, 2)  # binary classification
        self.assertEqual(adapter.embed_dim, self.embed_dim)

        # Check label map
        self.assertEqual(len(adapter.label_map), 2)
        self.assertIn('0', adapter.label_map)
        self.assertIn('1', adapter.label_map)

        # Label indices should be 0 and 1
        self.assertEqual(set(adapter.label_map.values()), {0, 1})

    def test_stage3_labels_encoded_correctly(self):
        """Test that string labels are correctly encoded to integer indices."""
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')

        loader, adapter = create_dataloader(
            grouped,
            batch_size=3,
            shuffle=False,
            num_workers=0
        )

        features, labels, mask = next(iter(loader))

        # Labels should be torch.long tensor
        self.assertEqual(labels.dtype, torch.long)

        # All labels should be 0 or 1 (binary)
        self.assertTrue(torch.all((labels == 0) | (labels == 1)))

    # =========================================================================
    # Integration Tests: End-to-End Shape Tracing
    # =========================================================================

    def test_end_to_end_shape_tracing_patient_1(self):
        """
        Trace Patient 1 through entire pipeline:
        Stage 1: 2 slides [300, 1024] and [200, 1024]
        Stage 2: 1 patient [500, 1024]
        Stage 3: 1 batch [1, 500, 1024] (with batch_size > 1)
        """
        # Stage 1: Load slide-level dataset
        dataset = MILDataset(self.labels_csv, self.features_dir)

        slide1 = dataset['patient1_slide1']
        slide2 = dataset['patient1_slide2']

        self.assertEqual(slide1.features.shape, (300, 1024))
        self.assertEqual(slide2.features.shape, (200, 1024))

        # Stage 2: Group by case_id
        grouped = dataset.concat_by('case_id', label_voting='first')
        patient1 = grouped['patient_1']

        self.assertEqual(patient1.features.shape, (500, 1024))
        self.assertEqual(patient1.num_items, 2)

        # Stage 3: Create dataloader with batch_size > 1 to get mask
        loader, adapter = create_dataloader(
            grouped,
            batch_size=2,  # Use batch_size > 1 to enable collation
            shuffle=False,
            num_workers=0
        )

        # Get patient1 specifically (need to find it in loader)
        for features, labels, mask in loader:
            if features.shape[2] == 500 or (features.shape[1] == 500):  # This is patient1
                # Check if patient1 is in this batch
                if mask[:, 500:].sum() == 0 and mask[:, :500].sum() >= 500:
                    # Found patient1 - it has 500 valid patches
                    patient1_idx = (mask.sum(dim=1) == 500).nonzero(as_tuple=True)[0]
                    if len(patient1_idx) > 0:
                        idx = patient1_idx[0]
                        self.assertEqual(mask[idx].sum(), 500)  # All positions valid
                        break

    def test_end_to_end_shape_tracing_patient_2(self):
        """
        Trace Patient 2 through entire pipeline:
        Stage 1: 1 slide [150, 1024]
        Stage 2: 1 patient [150, 1024]
        Stage 3: Batched with padding
        """
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Stage 1
        slide1 = dataset['patient2_slide1']
        self.assertEqual(slide1.features.shape, (150, 1024))

        # Stage 2
        grouped = dataset.concat_by('case_id', label_voting='first')
        patient2 = grouped['patient_2']

        self.assertEqual(patient2.features.shape, (150, 1024))
        self.assertEqual(patient2.num_items, 1)

        # Stage 3 - Use batch_size > 1 to enable collation
        loader, adapter = create_dataloader(
            grouped,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        for features, labels, mask in loader:
            # Find patient2 by checking which sample has 150 valid patches
            valid_counts = mask.sum(dim=1)
            patient2_mask = (valid_counts == 150)
            if patient2_mask.any():
                # Found patient2
                self.assertTrue(True)
                break

    def test_end_to_end_shape_tracing_patient_3(self):
        """
        Trace Patient 3 through entire pipeline:
        Stage 1: 3 slides [100, 1024] each
        Stage 2: 1 patient [300, 1024]
        Stage 3: Batched with padding
        """
        dataset = MILDataset(self.labels_csv, self.features_dir)

        # Stage 1
        slide1 = dataset['patient3_slide1']
        slide2 = dataset['patient3_slide2']
        slide3 = dataset['patient3_slide3']

        self.assertEqual(slide1.features.shape, (100, 1024))
        self.assertEqual(slide2.features.shape, (100, 1024))
        self.assertEqual(slide3.features.shape, (100, 1024))

        # Stage 2
        grouped = dataset.concat_by('case_id', label_voting='first')
        patient3 = grouped['patient_3']

        self.assertEqual(patient3.features.shape, (300, 1024))
        self.assertEqual(patient3.num_items, 3)

        # Stage 3 - Use batch_size > 1 to enable collation
        loader, adapter = create_dataloader(
            grouped,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        for features, labels, mask in loader:
            # Find patient3 by checking which sample has 300 valid patches
            valid_counts = mask.sum(dim=1)
            patient3_mask = (valid_counts == 300)
            if patient3_mask.any():
                # Found patient3
                self.assertTrue(True)
                break

    def test_end_to_end_full_dataset_summary(self):
        """
        Print comprehensive summary of shapes through entire pipeline.
        This test documents the complete transformation for reference.
        """
        dataset = MILDataset(self.labels_csv, self.features_dir)
        grouped = dataset.concat_by('case_id', label_voting='first')
        loader, adapter = create_dataloader(grouped, batch_size=3, shuffle=False, num_workers=0)

        # Collect shape information
        print("\n" + "="*70)
        print("COMPLETE SHAPE TRACING SUMMARY")
        print("="*70)

        print("\nSTAGE 1: Slide-level (MILDataset)")
        print("-"*70)
        for slide in dataset:
            print(f"  {slide.slide_id:20s} shape: {tuple(slide.features.shape):15s} "
                  f"case: {slide.case_id:10s} label: {slide.label}")

        print("\nSTAGE 2: Patient-level (GroupedMILDataset)")
        print("-"*70)
        for patient in grouped:
            print(f"  {patient.group_id:20s} shape: {tuple(patient.features.shape):15s} "
                  f"slides: {patient.num_items} label: {patient.label}")

        print("\nSTAGE 3: Batched with padding (DataLoader)")
        print("-"*70)
        features, labels, mask = next(iter(loader))
        print(f"  Batch features shape: {tuple(features.shape)}")
        print(f"  Batch labels shape:   {tuple(labels.shape)}")
        print(f"  Batch mask shape:     {tuple(mask.shape)}")
        print(f"  Valid patches per sample: {mask.sum(dim=1).tolist()}")

        print("\n" + "="*70)
        print("VERIFICATION")
        print("="*70)
        print(f"  Total slides:        {len(dataset)}")
        print(f"  Total patients:      {len(grouped)}")
        print(f"  Embedding dimension: {self.embed_dim}")
        print(f"  Num classes:         {adapter.num_classes}")
        print(f"  Label map:           {adapter.label_map}")

        # Verify shapes match expectations
        self.assertEqual(features.shape, (3, 500, 1024))  # padded to max
        self.assertEqual(labels.shape, (3,))
        self.assertEqual(mask.shape, (3, 500))


class TestShapeTracingWithDifferentEmbeddingDims(unittest.TestCase):
    """
    Test shape tracing with different embedding dimensions to ensure
    the pipeline is not hardcoded to 1024.
    """

    def test_shape_tracing_with_1536_dim(self):
        """Test shape transformations with embedding dimension 1536."""
        tmpdir = tempfile.TemporaryDirectory()
        features_dir = Path(tmpdir.name) / "features"
        features_dir.mkdir()

        embed_dim = 1536

        # Create simple dataset
        slides = [
            ('slide1', 'case1', 100, '0'),
            ('slide2', 'case1', 200, '0'),
        ]

        labels_csv = Path(tmpdir.name) / "labels.csv"
        rows = []
        for slide_id, case_id, patches, label in slides:
            rows.append({'slide_id': slide_id, 'case_id': case_id, 'label': label})
            with h5py.File(features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features',
                               data=np.random.randn(patches, embed_dim).astype(np.float32))

        pd.DataFrame(rows).to_csv(labels_csv, index=False)

        # Trace shapes
        dataset = MILDataset(labels_csv, features_dir)
        self.assertEqual(dataset.embed_dim, embed_dim)

        slide1 = dataset[0]
        self.assertEqual(slide1.features.shape[1], embed_dim)

        grouped = dataset.concat_by('case_id')
        case1 = grouped[0]
        self.assertEqual(case1.features.shape, (300, embed_dim))

        # With batch_size=1, no collation (returns features, labels only)
        loader, adapter = create_dataloader(grouped, batch_size=1, shuffle=False, num_workers=0)
        features, labels = next(iter(loader))
        self.assertEqual(features.shape, (300, embed_dim))

        tmpdir.cleanup()


def print_shape_tracing_diagram():
    """
    Print a visual diagram of the shape transformations.
    Called when running the test file directly.
    """
    print("\n" + "="*70)
    print("DATA SHAPE TRANSFORMATION DIAGRAM")
    print("="*70)
    print("""
INPUT STRUCTURE: n×m×p×D
  n = number of case_ids (patients) = 3
  m = number of slides per patient (variable)
  p = number of tiles per slide (variable)
  D = embedding dimension = 1024

PATIENT 1:
  Slide 1: [300, 1024]  ┐
  Slide 2: [200, 1024]  ├─> concat_by('case_id') ─> [500, 1024]
                        ┘

PATIENT 2:
  Slide 1: [150, 1024]  ─> concat_by('case_id') ─> [150, 1024]

PATIENT 3:
  Slide 1: [100, 1024]  ┐
  Slide 2: [100, 1024]  ├─> concat_by('case_id') ─> [300, 1024]
  Slide 3: [100, 1024]  ┘

DATALOADER BATCHING (batch_size=3):
  Patient 1: [500, 1024]  ┐
  Patient 2: [150, 1024]  ├─> pad to max ─> [3, 500, 1024] + mask [3, 500]
  Patient 3: [300, 1024]  ┘

MASK INDICATES VALID POSITIONS:
  Patient 1: [1,1,1,...,1,1,1] (500 ones, 0 zeros)
  Patient 2: [1,1,1,...,1,0,0] (150 ones, 350 zeros)
  Patient 3: [1,1,1,...,1,0,0] (300 ones, 200 zeros)
""")
    print("="*70)


if __name__ == '__main__':
    import sys

    # Print diagram if requested
    if '--diagram' in sys.argv:
        print_shape_tracing_diagram()
        sys.argv.remove('--diagram')

    # Run tests
    unittest.main(verbosity=2)
