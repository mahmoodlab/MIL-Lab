import pandas as pd
import numpy as np
import argparse
import json
import os
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Dict, Optional

def generate_splits(
    label_csv: str,
    output_dir: str,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    seed: int = 42,
    case_col: str = 'case_id',
    label_col: str = 'label'
):
    """
    Generate N-fold patient-level cross-validation splits.
    
    Args:
        label_csv: Path to CSV containing case labels.
        output_dir: Directory to save JSON split files.
        n_folds: Number of folds.
        val_ratio: Fraction of training set to use for validation.
        seed: Random seed.
        case_col: Column name for patient/case ID.
        label_col: Column name for label (used for StratifiedKFold if possible).
    """
    print(f"Generating {n_folds} folds from {label_csv}...")
    df = pd.read_csv(label_csv)
    
    if case_col not in df.columns:
        raise ValueError(f"Column '{case_col}' not found in CSV.")
        
    # Group by case_id to ensure patient-level splitting
    # We need one label per case for stratification
    # Assuming all slides for a case have the same label (standard in MIL)
    case_df = df.groupby(case_col)[label_col].first().reset_index()
    unique_cases = case_df[case_col].values
    case_labels = case_df[label_col].values
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Outer loop: K-Fold for Train/Test split
    # Using StratifiedKFold to maintain class balance
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(unique_cases, case_labels)):
        train_val_cases = unique_cases[train_val_indices]
        test_cases = unique_cases[test_indices]
        train_val_labels = case_labels[train_val_indices]
        
        # Inner split: Train/Val
        # We perform a single split on the train_val set
        # Adjust n_splits based on ratio: 1/ratio approximately
        # But a simpler way is `train_test_split` logic, but let's stick to KFold logic or manual shuffle
        # For reproducibility and simplicity with ratios, we can use a seeded shuffle
        
        rng = np.random.RandomState(seed + fold_idx) # Vary seed slightly per fold for inner split
        n_train_val = len(train_val_cases)
        n_val = int(n_train_val * val_ratio)
        if n_val == 0 and val_ratio > 0 and n_train_val >= 2:
            n_val = 1
        
        # Shuffle indices
        indices = np.arange(n_train_val)
        rng.shuffle(indices)
        
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        val_cases = train_val_cases[val_idx]
        train_cases = train_val_cases[train_idx]
        
        split_dict = {
            'train': train_cases.tolist(),
            'val': val_cases.tolist(),
            'test': test_cases.tolist(),
            'metadata': {
                'seed': seed,
                'fold': fold_idx,
                'source': label_csv,
                'val_ratio': val_ratio
            }
        }
        
        output_path = os.path.join(output_dir, f'splits_{fold_idx}.json')
        with open(output_path, 'w') as f:
            json.dump(split_dict, f, indent=4)
            
        print(f"Fold {fold_idx}: Train={len(train_cases)}, Val={len(val_cases)}, Test={len(test_cases)} cases. Saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patient-level K-Fold splits')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to labels CSV')
    parser.add_argument('--output_dir', type=str, default='splits', help='Output directory for JSONs')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio (fraction of train set)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    generate_splits(
        label_csv=args.label_csv,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
