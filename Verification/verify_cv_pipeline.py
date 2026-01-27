import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
import os
import tempfile
import h5py
from pathlib import Path
import subprocess

def verify_pipeline():
    print("Starting CV pipeline verification...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create dummy data
        csv_path = os.path.join(tmp_dir, 'labels.csv')
        feats_dir = os.path.join(tmp_dir, 'features')
        os.makedirs(feats_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'slide_id': [f'slide_{i}' for i in range(10)],
            'case_id': [f'case_{i//2}' for i in range(10)], # 5 cases
            'label': [0, 1] * 5
        })
        df.to_csv(csv_path, index=False)
        
        for slide_id in df.slide_id:
            with h5py.File(os.path.join(feats_dir, f'{slide_id}.h5'), 'w') as f:
                f.create_dataset('features', data=torch.randn(10, 1536).numpy()) # dim 1536
        
        # 2. Generate splits
        split_dir = os.path.join(tmp_dir, 'splits')
        cmd_gen = [
            'python3', 'scripts/create_splits.py',
            '--label_csv', csv_path,
            '--output_dir', split_dir,
            '--n_folds', '2',
            '--val_ratio', '0.5'
        ]
        subprocess.run(cmd_gen, check=True)
        
        # 3. Create config
        config_path = os.path.join(tmp_dir, 'config.json')
        config = {
            "data": {
                "labels_csv": csv_path,
                "features_dir": feats_dir,
                "num_workers": 0,
                "hierarchical": False,
                "group_column": "case_id"
            },
            "train": {
                "num_epochs": 1,
                "min_epochs": 1,
                "batch_size": 1,
                "learning_rate": 0.001,
                "use_amp": False,
                "early_stopping_patience": 5,
                "task_type": "binary"
            },
            "model_name": "abmil.base.uni_v2.none", # Need a real model name that works with create_model
            "num_classes": 2,
            "output_dir": os.path.join(tmp_dir, "output")
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # 4. Run training for Fold 0
        cmd_train = [
            'python3', 'run_mil_experiments_predefined_splits.py',
            '--config', config_path,
            '--split_dir', split_dir,
            '--fold', '0'
        ]
        # We need to make sure src and training modules are in path
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(cmd_train, check=True, env=env)
        
        print("CV Pipeline verification successful!")

if __name__ == "__main__":
    verify_pipeline()
