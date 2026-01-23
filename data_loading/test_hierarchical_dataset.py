import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
from data_loading.dataset import MILDataset, HierarchicalMILDataset

class TestHierarchicalDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.tmpdir.name) / "features"
        self.features_dir.mkdir()
        
        # Create mock slides
        # case_A: 2 slides
        # case_B: 1 slide
        self.slide_data = {
            'slide_001': {'case_id': 'case_A', 'patches': 10, 'label': 1},
            'slide_002': {'case_id': 'case_A', 'patches': 20, 'label': 1},
            'slide_003': {'case_id': 'case_B', 'patches': 15, 'label': 0},
        }
        
        self.labels_csv = Path(self.tmpdir.name) / "labels.csv"
        rows = []
        for slide_id, info in self.slide_data.items():
            rows.append({
                'slide_id': slide_id,
                'case_id': info['case_id'],
                'label': info['label']
            })
            
            # Create H5 file
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(info['patches'], 128).astype(np.float32))
        
        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)
        self.base_dataset = MILDataset(self.labels_csv, self.features_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_hierarchical_grouping(self):
        """Test that group_by creates a HierarchicalMILDataset with correct structure."""
        hier_dataset = self.base_dataset.group_by('case_id')
        
        self.assertIsInstance(hier_dataset, HierarchicalMILDataset)
        self.assertEqual(len(hier_dataset), 2)  # case_A, case_B
        
        # Check case_A
        case_a = hier_dataset['case_A']
        self.assertEqual(len(case_a.features), 2)
        self.assertEqual(case_a.features[0].shape, (10, 128))
        self.assertEqual(case_a.features[1].shape, (20, 128))
        self.assertEqual(case_a.label, '1')
        
        # Check case_B
        case_b = hier_dataset['case_B']
        self.assertEqual(len(case_b.features), 1)
        self.assertEqual(case_b.features[0].shape, (15, 128))
        self.assertEqual(case_b.label, '0')

    def test_to_padded_tensor(self):
        """Test conversion to padded tensor for hierarchical MIL."""
        hier_dataset = self.base_dataset.group_by('case_id')
        case_a = hier_dataset['case_A']
        
        padded, mask = case_a.to_padded_tensor()
        
        # 2 slides, max patches = 20
        self.assertEqual(padded.shape, (2, 20, 128))
        self.assertEqual(mask.shape, (2, 20))
        
        # Check first slide (10 patches)
        self.assertTrue(torch.all(mask[0, :10] == 1.0))
        self.assertTrue(torch.all(mask[0, 10:] == 0.0))
        
        # Check second slide (20 patches)
        self.assertTrue(torch.all(mask[1, :] == 1.0))

    def test_random_split_no_leakage(self):
        """Test that random_split on HierarchicalMILDataset doesn't leak slides from the same case."""
        # Create a larger dataset to test splitting
        rows = []
        for i in range(20):
            case_id = f"case_{i // 2}" # 2 slides per case
            slide_id = f"slide_{i:03d}"
            rows.append({'slide_id': slide_id, 'case_id': case_id, 'label': i % 2})
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(5, 128).astype(np.float32))
        
        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)
        dataset = MILDataset(self.labels_csv, self.features_dir)
        hier_dataset = dataset.group_by('case_id')
        
        splits = hier_dataset.random_split(train_frac=0.6, val_frac=0.2, seed=42)
        
        train_cases = set(splits['train'].group_ids)
        val_cases = set(splits['val'].group_ids)
        test_cases = set(splits['test'].group_ids)
        
        # Ensure no overlap between case sets
        self.assertTrue(train_cases.isdisjoint(val_cases))
        self.assertTrue(train_cases.isdisjoint(test_cases))
        self.assertTrue(val_cases.isdisjoint(test_cases))
        
        # Verify slides within each split belong ONLY to those cases
        for split_name in ['train', 'val', 'test']:
            split_ds = splits[split_name]
            for group in split_ds:
                case_id = group.group_id
                self.assertIn(case_id, getattr(self, f"{split_name}_cases", set(split_ds.group_ids)))

if __name__ == '__main__':
    unittest.main()
