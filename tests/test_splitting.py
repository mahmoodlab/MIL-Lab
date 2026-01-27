import unittest
import pandas as pd
import tempfile
import json
import os
from pathlib import Path
import sys

# Add project root to path to import scripts
sys.path.append(str(Path(__file__).parents[1]))

from scripts.create_splits import generate_splits

class TestSplitting(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'dummy_labels.csv')
        self.output_dir = os.path.join(self.temp_dir.name, 'splits')
        os.makedirs(self.output_dir, exist_ok=True)

        data = {
            'slide_id': [f'slide_{i}' for i in range(20)],
            'case_id': [f'case_{i//2}' for i in range(20)], # 10 cases, 2 slides each
            'label': [0, 1] * 10
        }
        pd.DataFrame(data).to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_integrity(self):
        """Verify structure and non-overlap."""
        n_folds = 5
        generate_splits(
            label_csv=self.csv_path,
            output_dir=self.output_dir,
            n_folds=n_folds,
            val_ratio=0.2,
            seed=42
        )

        for i in range(n_folds):
            split_file = os.path.join(self.output_dir, f'splits_{i}.json')
            self.assertTrue(os.path.exists(split_file))

            with open(split_file, 'r') as f:
                split = json.load(f)

            train = set(split['train'])
            val = set(split['val'])
            test = set(split['test'])

            # Check for overlap
            self.assertTrue(train.isdisjoint(val))
            self.assertTrue(train.isdisjoint(test))
            self.assertTrue(val.isdisjoint(test))

            # Check coverage (should cover all 10 cases)
            all_cases = train | val | test
            self.assertEqual(len(all_cases), 10)

    def test_reproducibility(self):
        """Verify same seed produces same splits."""
        generate_splits(self.csv_path, self.output_dir, n_folds=2, seed=42)
        with open(os.path.join(self.output_dir, 'splits_0.json'), 'r') as f:
            run1 = json.load(f)
        
        generate_splits(self.csv_path, self.output_dir, n_folds=2, seed=42)
        with open(os.path.join(self.output_dir, 'splits_0.json'), 'r') as f:
            run2 = json.load(f)
            
        self.assertEqual(run1, run2)

if __name__ == '__main__':
    unittest.main()
