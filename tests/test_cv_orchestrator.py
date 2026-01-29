import unittest
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from scripts.run_cv_orchestrator import find_folds, aggregate_results, run_fold_worker

class TestCVOrchestrator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.split_dir = os.path.join(self.temp_dir.name, 'splits')
        os.makedirs(self.split_dir, exist_ok=True)
        
        # Create dummy splits
        for i in range(3):
            with open(os.path.join(self.split_dir, f'splits_{i}.json'), 'w') as f:
                json.dump({'test': [f'case_{i}']}, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_folds(self):
        folds = find_folds(self.split_dir)
        self.assertEqual(folds, [0, 1, 2])

    @patch('subprocess.run')
    def test_run_fold_worker(self, mock_run):
        mock_run.return_value = MagicMock(stdout='Success', stderr='', returncode=0)
        
        args = {
            'fold': 0,
            'config': 'config.json',
            'split_dir': self.split_dir,
            'output_dir': self.temp_dir.name
        }
        
        result = run_fold_worker(args)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['fold'], 0)
        self.assertTrue(mock_run.called)

    def test_aggregate_results(self):
        # Create dummy results
        output_dir = os.path.join(self.temp_dir.name, 'output')
        for i in range(3):
            fold_dir = os.path.join(output_dir, f'fold_{i}')
            os.makedirs(fold_dir, exist_ok=True)
            results = {
                'accuracy': 0.8 + i * 0.05,
                'quadratic_kappa': 0.7 + i * 0.05
            }
            with open(os.path.join(fold_dir, 'test_results.json'), 'w') as f:
                json.dump(results, f)
        
        df = aggregate_results(output_dir)
        self.assertEqual(len(df), 3)
        self.assertIn('accuracy', df.columns)
        self.assertIn('quadratic_kappa', df.columns)
        
        # Check mean calculation
        self.assertAlmostEqual(df['accuracy'].mean(), 0.85) # (0.8 + 0.85 + 0.9) / 3
        
        # Check CSV existence
        self.assertTrue(os.path.exists(os.path.join(output_dir, 'cv_summary.csv')))

if __name__ == '__main__':
    unittest.main()
