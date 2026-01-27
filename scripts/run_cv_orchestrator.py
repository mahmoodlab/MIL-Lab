#!/usr/bin/env python3
"""
CV Orchestrator
Executes K-fold cross-validation in parallel and aggregates results.
"""

import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
import subprocess
import multiprocessing
from typing import List, Dict, Any
from pathlib import Path

def find_folds(split_dir: str) -> List[int]:
    """Find available fold indices in split directory."""
    split_files = glob.glob(os.path.join(split_dir, 'splits_*.json'))
    folds = []
    for f in split_files:
        try:
            # Extract number from filename "splits_N.json"
            basename = os.path.basename(f)
            fold_num = int(basename.replace('splits_', '').replace('.json', ''))
            folds.append(fold_num)
        except ValueError:
            continue
    return sorted(folds)

def run_fold_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to run a single fold. 
    
    Args:
        args: Dictionary containing:
            - fold: int
            - config: str (path)
            - split_dir: str (path)
            - output_dir: str (path)
            
    Returns:
        Dictionary with status and result path
    """
    fold = args['fold']
    print(f"[Fold {fold}] Starting...")
    
    cmd = [
        'python3', 'run_mil_experiments_predefined_splits.py',
        '--config', args['config'],
        '--split_dir', args['split_dir'],
        '--fold', str(fold)
    ]
    
    # Override output directory if provided to orchestrator
    if args.get('output_dir'):
        cmd.extend(['--output_dir', args['output_dir']])
        fold_output_dir = os.path.join(args['output_dir'], f'fold_{fold}')
    
    # Ensure PYTHONPATH includes current directory so modules can be imported
    env = os.environ.copy()
    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.getcwd()
    else:
        env['PYTHONPATH'] = os.getcwd() + os.pathsep + env['PYTHONPATH']

    try:
        # Capture output to avoid interleaved printing
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            env=env
        )
        print(f"[Fold {fold}] Completed successfully.")
        return {'fold': fold, 'status': 'success', 'output_dir': fold_output_dir if args.get('output_dir') else None}
    except subprocess.CalledProcessError as e:
        print(f"[Fold {fold}] FAILED.")
        print(f"[Fold {fold}] Error: {e.stderr}")
        return {'fold': fold, 'status': 'failed', 'error': e.stderr}

def aggregate_results(output_dir: str) -> pd.DataFrame:
    """
    Aggregate results from all folds.
    """
    results_files = glob.glob(os.path.join(output_dir, 'fold_*', 'test_results.json'))
    
    all_metrics = []
    
    for res_file in results_files:
        try:
            with open(res_file, 'r') as f:
                data = json.load(f)
                
            # Extract fold number from path if possible, or just add it
            fold_dir = os.path.basename(os.path.dirname(res_file))
            fold_num = int(fold_dir.replace('fold_', ''))
            
            # Flatten metrics
            metrics = {'fold': fold_num}
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    metrics[k] = v
            
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error reading {res_file}: {e}")
            
    if not all_metrics:
        print("No results found to aggregate.")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_metrics).sort_values('fold')
    
    # Calculate summary stats
    summary = df.describe().transpose()[['mean', 'std', 'min', 'max']]
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(summary)
    
    # Save detailed CSV
    summary_path = os.path.join(output_dir, 'cv_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nDetailed results saved to: {summary_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='CV Orchestrator')
    parser.add_argument('--config', type=str, required=True, help='Path to ExperimentConfig')
    parser.add_argument('--split_dir', type=str, required=True, help='Directory containing split JSONs')
    parser.add_argument('--output_dir', type=str, required=True, help='Root output directory for CV run')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Find folds
    folds = find_folds(args.split_dir)
    if not folds:
        print(f"No splits found in {args.split_dir}")
        return
        
    print(f"Found {len(folds)} folds: {folds}")
    print(f"Running with {args.workers} workers...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare worker args
    worker_args = []
    for fold in folds:
        worker_args.append({
            'fold': fold,
            'config': args.config,
            'split_dir': args.split_dir,
            'output_dir': args.output_dir
        })
        
    # Run in parallel
    if args.workers > 1:
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = pool.map(run_fold_worker, worker_args)
    else:
        # Sequential for debugging or single worker
        results = [run_fold_worker(arg) for arg in worker_args]
        
    # Check failures
    failed = [r for r in results if r['status'] == 'failed']
    if failed:
        print(f"\n{len(failed)} folds failed!")
    else:
        print("\nAll folds completed successfully.")
        
    # Aggregate
    aggregate_results(args.output_dir)

if __name__ == '__main__':
    main()
