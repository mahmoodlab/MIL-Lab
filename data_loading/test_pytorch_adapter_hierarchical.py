import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
import h5py
from pathlib import Path
from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader

class TestPytorchAdapter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.features_dir = Path(self.tmpdir.name) / "features"
        self.features_dir.mkdir()
        
        # Create mock slides
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
            with h5py.File(self.features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(info['patches'], 128).astype(np.float32))
        
        pd.DataFrame(rows).to_csv(self.labels_csv, index=False)
        self.base_dataset = MILDataset(self.labels_csv, self.features_dir)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_hierarchical_dataloader(self):
        """Test that create_dataloader works with HierarchicalMILDataset."""
        hier_dataset = self.base_dataset.group_by('case_id')
        loader, adapter = create_dataloader(hier_dataset, batch_size=1, shuffle=False)
        
        self.assertEqual(len(loader), 2)
        
        batches = list(loader)
        # Batch 1: case_A
        features_lists, labels, masks = batches[0]
        # hierarchical_collate_fn returns list of lists for features
        self.assertEqual(len(features_lists), 1) # batch_size=1
        self.assertEqual(len(features_lists[0]), 2) # case_A has 2 slides
        self.assertEqual(features_lists[0][0].shape, (10, 128))
        self.assertEqual(features_lists[0][1].shape, (20, 128))
        
        # Batch 2: case_B
        features_lists, labels, masks = batches[1]
        self.assertEqual(len(features_lists[0]), 1) # case_B has 1 slide

if __name__ == '__main__':
    unittest.main()
