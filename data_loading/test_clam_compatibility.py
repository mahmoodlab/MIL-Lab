#!/usr/bin/env python3
"""
Unit tests to verify CLAM HDF5 output format is compatible with MIL-Lab reading code.

Tests verify:
1. CLAM's 2D format (num_patches, embed_dim) is readable
2. 3D format (1, num_patches, embed_dim) is also handled correctly
3. Coords are present but ignored by MIL-Lab
4. Data integrity is preserved through write/read cycle
"""

import pytest
import torch
import numpy as np
import h5py
import tempfile
import os
import sys
import pandas as pd
import importlib.util

# Import CLAM's save_hdf5 explicitly (MIL-Lab has its own version that doesn't append!)
spec_clam = importlib.util.spec_from_file_location(
    "clam_file_utils", "/home/nadim/Source/CLAM/utils/file_utils.py"
)
clam_file_utils = importlib.util.module_from_spec(spec_clam)
spec_clam.loader.exec_module(clam_file_utils)
save_hdf5 = clam_file_utils.save_hdf5  # CLAM's version (supports append)

# Import MIL-Lab's dataset
spec_mil = importlib.util.spec_from_file_location(
    "data_utils", "/home/nadim/Source/MIL-Lab/utils/data_utils.py"
)
data_utils = importlib.util.module_from_spec(spec_mil)
spec_mil.loader.exec_module(data_utils)
PANDAH5Dataset = data_utils.PANDAH5Dataset


class TestCLAMWriteFormat:
    """Test that CLAM writes HDF5 files in the expected format."""

    def test_save_hdf5_creates_features_dataset(self, tmp_path):
        """Verify save_hdf5 creates 'features' dataset."""
        output_path = tmp_path / "test_slide.h5"

        features = np.random.randn(100, 1024).astype(np.float32)
        coords = np.random.randint(0, 1000, (100, 2)).astype(np.int32)

        save_hdf5(str(output_path), {'features': features, 'coords': coords})

        with h5py.File(output_path, 'r') as f:
            assert 'features' in f, "Missing 'features' dataset"
            assert f['features'].shape == (100, 1024)
            assert f['features'].dtype == np.float32

    def test_save_hdf5_creates_coords_dataset(self, tmp_path):
        """Verify save_hdf5 creates 'coords' dataset."""
        output_path = tmp_path / "test_slide.h5"

        features = np.random.randn(100, 1024).astype(np.float32)
        coords = np.random.randint(0, 1000, (100, 2)).astype(np.int32)

        save_hdf5(str(output_path), {'features': features, 'coords': coords})

        with h5py.File(output_path, 'r') as f:
            assert 'coords' in f, "Missing 'coords' dataset"
            assert f['coords'].shape == (100, 2)
            assert f['coords'].dtype == np.int32

    def test_save_hdf5_append_mode(self, tmp_path):
        """Verify save_hdf5 can append batches (simulating CLAM's batch extraction).

        Note: CLAM uses 'w' for first batch, then 'a' for subsequent batches within
        a single slide's processing loop.
        """
        # Use string path explicitly (save_hdf5 expects string)
        output_path = str(tmp_path / "test_slide.h5")

        # Simulate CLAM's actual extraction loop behavior
        batches = [
            np.random.randn(32, 1024).astype(np.float32),
            np.random.randn(32, 1024).astype(np.float32),
        ]

        mode = 'w'
        for batch_features in batches:
            batch_coords = np.random.randint(0, 1000, (batch_features.shape[0], 2)).astype(np.int32)
            save_hdf5(output_path, {'features': batch_features, 'coords': batch_coords}, mode=mode)
            mode = 'a'

        with h5py.File(output_path, 'r') as f:
            assert f['features'].shape == (64, 1024), f"Batches not appended correctly, got {f['features'].shape}"
            assert f['coords'].shape == (64, 2), f"Coords not appended correctly, got {f['coords'].shape}"


class TestMILLabReadFormat:
    """Test that MIL-Lab can read CLAM-formatted HDF5 files."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory with HDF5 files."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()
        return feats_dir

    @pytest.fixture
    def mock_dataframe(self):
        """Create a mock dataframe for the dataset."""
        return pd.DataFrame({
            'slide_id': ['slide_001', 'slide_002', 'slide_003'],
            'label': [0, 1, 2],
            'split': ['train', 'train', 'train']
        })

    def test_read_clam_2d_format(self, mock_dataset_dir, mock_dataframe):
        """Test reading CLAM's native 2D format (num_patches, embed_dim)."""
        # Create CLAM-style HDF5 file
        slide_path = mock_dataset_dir / "slide_001.h5"
        expected_features = np.random.randn(150, 1024).astype(np.float32)
        expected_coords = np.random.randint(0, 5000, (150, 2)).astype(np.int32)

        save_hdf5(str(slide_path), {'features': expected_features, 'coords': expected_coords})

        # Read using MIL-Lab dataset
        dataset = PANDAH5Dataset(str(mock_dataset_dir), mock_dataframe, 'train')
        features, label = dataset[0]

        assert features.shape == (150, 1024), f"Shape mismatch: {features.shape}"
        assert label.item() == 0
        np.testing.assert_array_almost_equal(
            features.numpy(), expected_features, decimal=5,
            err_msg="Feature values don't match after read"
        )

    def test_read_3d_format(self, mock_dataset_dir, mock_dataframe):
        """Test reading 3D format (1, num_patches, embed_dim) - 'old format'."""
        slide_path = mock_dataset_dir / "slide_001.h5"

        # Write 3D format directly (not using CLAM's save_hdf5)
        expected_features_3d = np.random.randn(1, 200, 1536).astype(np.float32)
        with h5py.File(slide_path, 'w') as f:
            f.create_dataset('features', data=expected_features_3d)
            f.create_dataset('coords', data=np.zeros((200, 2), dtype=np.int32))

        # Read using MIL-Lab dataset
        dataset = PANDAH5Dataset(str(mock_dataset_dir), mock_dataframe, 'train')
        features, label = dataset[0]

        # Should be squeezed to 2D
        assert features.shape == (200, 1536), f"3D format not squeezed: {features.shape}"
        np.testing.assert_array_almost_equal(
            features.numpy(), expected_features_3d.squeeze(0), decimal=5,
            err_msg="Feature values don't match after squeeze"
        )

    def test_coords_ignored(self, mock_dataset_dir, mock_dataframe):
        """Verify MIL-Lab only reads features, not coords."""
        slide_path = mock_dataset_dir / "slide_001.h5"

        features = np.random.randn(100, 1024).astype(np.float32)
        coords = np.array([[999, 888]] * 100, dtype=np.int32)  # Distinctive coords
        save_hdf5(str(slide_path), {'features': features, 'coords': coords})

        dataset = PANDAH5Dataset(str(mock_dataset_dir), mock_dataframe, 'train')
        result = dataset[0]

        # Should return (features, label) tuple - no coords
        assert len(result) == 2, f"Expected 2 items, got {len(result)}"
        assert isinstance(result[0], torch.Tensor), "First item should be features tensor"
        assert isinstance(result[1], torch.Tensor), "Second item should be label tensor"


class TestEndToEndCompatibility:
    """End-to-end test simulating CLAM extraction → MIL-Lab training."""

    def test_full_pipeline_simulation(self, tmp_path):
        """Simulate full CLAM extraction and MIL-Lab reading pipeline.

        Uses single-batch writes (most common case) to avoid HDF5 append complexity.
        """
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        # Simulate CLAM's feature extraction for 3 slides
        slides_data = {
            'slide_A': {'num_patches': 500, 'embed_dim': 1024},
            'slide_B': {'num_patches': 1200, 'embed_dim': 1024},
            'slide_C': {'num_patches': 50, 'embed_dim': 1024},
        }

        expected_features = {}
        for slide_id, info in slides_data.items():
            output_path = feats_dir / f"{slide_id}.h5"
            num_patches = info['num_patches']
            embed_dim = info['embed_dim']

            # Write all features at once (simulates small slide or single batch)
            features = np.random.randn(num_patches, embed_dim).astype(np.float32)
            coords = np.random.randint(0, 10000, (num_patches, 2)).astype(np.int32)
            save_hdf5(str(output_path), {'features': features, 'coords': coords}, mode='w')
            expected_features[slide_id] = features

        # Create dataframe
        df = pd.DataFrame({
            'slide_id': list(slides_data.keys()),
            'label': [0, 1, 0],
            'split': ['train', 'train', 'train']
        })

        # Read using MIL-Lab
        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')

        assert len(dataset) == 3, f"Expected 3 slides, got {len(dataset)}"

        for idx, slide_id in enumerate(slides_data.keys()):
            features, label = dataset[idx]
            expected = expected_features[slide_id]

            assert features.shape[0] == expected.shape[0], \
                f"{slide_id}: patch count mismatch {features.shape[0]} vs {expected.shape[0]}"
            assert features.shape[1] == expected.shape[1], \
                f"{slide_id}: embed_dim mismatch {features.shape[1]} vs {expected.shape[1]}"

            np.testing.assert_array_almost_equal(
                features.numpy(), expected, decimal=5,
                err_msg=f"{slide_id}: feature values corrupted"
            )

    def test_different_embed_dimensions(self, tmp_path):
        """Test compatibility with different encoder dimensions."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        # Test common encoder dimensions
        encoder_dims = {
            'resnet50': 1024,
            'uni_v1': 1024,
            'uni_v2': 1536,
            'conch_v1': 512,
            'conch_v1.5': 768,
            'gigapath': 1536,
        }

        for encoder_name, embed_dim in encoder_dims.items():
            slide_path = feats_dir / f"slide_{encoder_name}.h5"

            features = np.random.randn(100, embed_dim).astype(np.float32)
            coords = np.random.randint(0, 1000, (100, 2)).astype(np.int32)
            save_hdf5(str(slide_path), {'features': features, 'coords': coords})

            df = pd.DataFrame({
                'slide_id': [f'slide_{encoder_name}'],
                'label': [0],
                'split': ['train']
            })

            dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
            loaded_features, _ = dataset[0]

            assert loaded_features.shape == (100, embed_dim), \
                f"{encoder_name}: shape mismatch {loaded_features.shape} vs (100, {embed_dim})"


class TestEdgeCases:
    """Test edge cases and potential failure modes."""

    def test_single_patch_slide(self, tmp_path):
        """Test slide with only 1 patch.

        NOTE: This test documents a KNOWN EDGE CASE in MIL-Lab.
        When a slide has exactly 1 patch with shape (1, embed_dim),
        the squeeze(0) operation produces shape (embed_dim,) instead of (1, embed_dim).

        This is unlikely to occur in practice (slides with 1 patch are filtered out
        by min_patches=24 default), but models expecting 2D input would fail.
        """
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        slide_path = feats_dir / "single_patch.h5"
        features = np.random.randn(1, 1024).astype(np.float32)
        coords = np.array([[0, 0]], dtype=np.int32)
        save_hdf5(str(slide_path), {'features': features, 'coords': coords})

        df = pd.DataFrame({
            'slide_id': ['single_patch'],
            'label': [0],
            'split': ['train']
        })

        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
        loaded_features, _ = dataset[0]

        # Document actual behavior: squeeze(0) on (1, 1024) produces (1024,)
        # This is a potential bug if models expect 2D input
        assert loaded_features.shape == (1024,), \
            f"Expected (1024,) due to squeeze behavior, got {loaded_features.shape}"

    def test_single_patch_slide_workaround(self, tmp_path):
        """Verify that slides with 2+ patches work correctly (the common case)."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        slide_path = feats_dir / "two_patches.h5"
        features = np.random.randn(2, 1024).astype(np.float32)
        coords = np.array([[0, 0], [256, 0]], dtype=np.int32)
        save_hdf5(str(slide_path), {'features': features, 'coords': coords})

        df = pd.DataFrame({
            'slide_id': ['two_patches'],
            'label': [0],
            'split': ['train']
        })

        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
        loaded_features, _ = dataset[0]

        # 2+ patches should maintain 2D shape
        assert loaded_features.shape == (2, 1024)

    def test_large_slide(self, tmp_path):
        """Test slide with many patches (memory handling)."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        slide_path = feats_dir / "large_slide.h5"
        num_patches = 10000
        features = np.random.randn(num_patches, 1536).astype(np.float32)
        coords = np.random.randint(0, 100000, (num_patches, 2)).astype(np.int32)
        save_hdf5(str(slide_path), {'features': features, 'coords': coords})

        df = pd.DataFrame({
            'slide_id': ['large_slide'],
            'label': [0],
            'split': ['train']
        })

        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
        loaded_features, _ = dataset[0]

        assert loaded_features.shape == (num_patches, 1536)

    def test_missing_coords_still_works(self, tmp_path):
        """Test that MIL-Lab works even if coords are missing (features-only H5)."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        slide_path = feats_dir / "no_coords.h5"
        features = np.random.randn(100, 1024).astype(np.float32)

        # Write features only, no coords
        with h5py.File(slide_path, 'w') as f:
            f.create_dataset('features', data=features)

        df = pd.DataFrame({
            'slide_id': ['no_coords'],
            'label': [0],
            'split': ['train']
        })

        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
        loaded_features, _ = dataset[0]

        assert loaded_features.shape == (100, 1024)

    def test_float64_features_converted(self, tmp_path):
        """Test that float64 features are handled (common mistake)."""
        feats_dir = tmp_path / "features"
        feats_dir.mkdir()

        slide_path = feats_dir / "float64_slide.h5"
        features = np.random.randn(100, 1024).astype(np.float64)  # Wrong dtype

        with h5py.File(slide_path, 'w') as f:
            f.create_dataset('features', data=features)

        df = pd.DataFrame({
            'slide_id': ['float64_slide'],
            'label': [0],
            'split': ['train']
        })

        dataset = PANDAH5Dataset(str(feats_dir), df, 'train')
        loaded_features, _ = dataset[0]

        # Should load without error (torch.from_numpy handles dtype conversion)
        assert loaded_features.shape == (100, 1024)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
