import torch
import torch.nn as nn
import pandas as pd
import json
import os
import tempfile
import h5py
from pathlib import Path
import subprocess
import shutil

def verify_orchestrator():
    print("Starting Orchestrator pipeline verification...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create dummy data
        csv_path = os.path.join(tmp_dir, 'labels.csv')
        feats_dir = os.path.join(tmp_dir, 'features')
        os.makedirs(feats_dir, exist_ok=True)
        
        df = pd.DataFrame({
            'slide_id': [f'slide_{i}' for i in range(40)],
            'case_id': [f'case_{i//2}' for i in range(40)], # 20 cases
            'label': [0, 1] * 20
        })
        df.to_csv(csv_path, index=False)
        
        for slide_id in df.slide_id:
            with h5py.File(os.path.join(feats_dir, f'{slide_id}.h5'), 'w') as f:
                f.create_dataset('features', data=torch.randn(10, 1536).numpy())
        
        # 2. Generate splits
        split_dir = os.path.join(tmp_dir, 'splits')
        cmd_gen = [
            'python3', 'scripts/create_splits.py',
            '--label_csv', csv_path,
            '--output_dir', split_dir,
            '--n_folds', '3', # 3 folds
            '--val_ratio', '0.2'
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
            "model_name": "abmil.base.uni_v2.none",
            "num_classes": 2,
            "output_dir": os.path.join(tmp_dir, "cv_output")
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # 4. Run Orchestrator
        output_dir = os.path.join(tmp_dir, "cv_output")
        cmd_orch = [
            'python3', 'scripts/run_cv_orchestrator.py',
            '--config', config_path,
            '--split_dir', split_dir,
            '--output_dir', output_dir,
            '--workers', '2'
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        subprocess.run(cmd_orch, check=True, env=env)
        
        # 5. Verify outputs
        summary_path = os.path.join(output_dir, 'cv_summary.csv')
        if os.path.exists(summary_path):
            print("Summary CSV found.")
            df_res = pd.read_csv(summary_path)
            print(f"Results for {len(df_res)} folds found.")
            if len(df_res) == 3:
                print("Orchestrator Pipeline verification successful!")
            else:
                print(f"Error: Expected 3 folds, found {len(df_res)}")
        else:
            print("Error: Summary CSV not found!")

if __name__ == "__main__":
    verify_orchestrator()
