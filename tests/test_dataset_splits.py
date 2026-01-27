import unittest
import pandas as pd
import tempfile
import json
import os
import torch
import h5py
import sys
from pathlib import Path

# Add project root to path to import data_loading
sys.path.append(str(Path(__file__).parents[1]))

from data_loading.dataset import MILDataset

class TestDatasetSplits(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'labels.csv')
        self.features_dir = os.path.join(self.temp_dir.name, 'features')
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Create dummy labels
        self.df = pd.DataFrame({
            'slide_id': [f'slide_{i}' for i in range(10)],
            'case_id': [f'case_{i//2}' for i in range(10)], # 5 cases
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        self.df.to_csv(self.csv_path, index=False)
        
        # Create dummy H5 files
        for slide_id in self.df.slide_id:
            with h5py.File(os.path.join(self.features_dir, f'{slide_id}.h5'), 'w') as f:
                f.create_dataset('features', data=torch.randn(10, 1536).numpy())
                
        # Create dummy split JSON
        self.split_json = os.path.join(self.temp_dir.name, 'split_0.json')
        self.split_dict = {
            'train': ['case_0', 'case_1'],
            'val': ['case_2'],
            'test': ['case_3', 'case_4']
        }
        with open(self.split_json, 'w') as f:
            json.dump(self.split_dict, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mil_dataset_load_split(self):
        dataset = MILDataset(self.csv_path, self.features_dir)
        
        train_ds = dataset.load_split(self.split_json, 'train')
        self.assertEqual(len(train_ds), 4) # case_0 and case_1 have 2 slides each
        self.assertTrue(all(c in ['case_0', 'case_1'] for c in train_ds.df.case_id))
        
        val_ds = dataset.load_split(self.split_json, 'val')
        self.assertEqual(len(val_ds), 2)
        self.assertTrue(all(c == 'case_2' for c in val_ds.df.case_id))

    def test_grouped_dataset_load_split(self):
        dataset = MILDataset(self.csv_path, self.features_dir)
        grouped_ds = dataset.concat_by('case_id')
        
        train_ds = grouped_ds.load_split(self.split_json, 'train')
        self.assertEqual(len(train_ds), 2) # 2 cases
        self.assertTrue(all(c in ['case_0', 'case_1'] for c in train_ds.group_ids))

if __name__ == '__main__':
    unittest.main()
