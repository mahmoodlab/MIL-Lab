#!/usr/bin/env python3
"""
CV Orchestrator
Executes K-fold cross-validation in parallel and aggregates results.

Usage:
    # Run 5-fold CV with 2 parallel workers
    python scripts/run_cv_orchestrator.py --config experiment.json --n_folds 5 --output_dir results/ --workers 2

    # Run sequentially (verbose output)
    python scripts/run_cv_orchestrator.py --config experiment.json --n_folds 5 --output_dir results/ --workers 1
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

def get_fold_indices(n_folds: int) -> List[int]:
    """Get list of fold indices (0-indexed)."""
    return list(range(n_folds))

def run_fold_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function to run a single fold.

    Args:
        args: Dictionary containing:
            - fold: int
            - config: str (path)
            - n_folds: int
            - output_dir: str (path)
            - verbose: bool (show real-time output)

    Returns:
        Dictionary with status and result path
    """
    fold = args['fold']
    verbose = args.get('verbose', False)
    print(f"[Fold {fold}] Starting...")

    cmd = [
        'python3', 'run_mil_experiments_cv.py',
        '--config', args['config'],
        '--n_folds', str(args['n_folds']),
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
        if verbose:
            # Show output in real-time (for single worker mode)
            result = subprocess.run(
                cmd,
                check=True,
                env=env
            )
        else:
            # Log output to file for parallel mode (allows tail -f)
            log_file = os.path.join(fold_output_dir, 'training.log') if args.get('output_dir') else f'fold_{fold}.log'
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            print(f"[Fold {fold}] Logging to: {log_file}")
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env
                )
        print(f"[Fold {fold}] Completed successfully.")
        return {'fold': fold, 'status': 'success', 'output_dir': fold_output_dir if args.get('output_dir') else None}
    except subprocess.CalledProcessError as e:
        print(f"[Fold {fold}] FAILED.")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"[Fold {fold}] Error: {e.stderr}")
        else:
            log_file = os.path.join(fold_output_dir, 'training.log') if args.get('output_dir') else f'fold_{fold}.log'
            print(f"[Fold {fold}] Check log: {log_file}")
        return {'fold': fold, 'status': 'failed', 'error': getattr(e, 'stderr', str(e))}

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
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--output_dir', type=str, required=True, help='Root output directory for CV run')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--quiet', action='store_true', help='Suppress real-time training output (auto-enabled for parallel runs)')

    args = parser.parse_args()

    # Get fold indices
    folds = get_fold_indices(args.n_folds)

    # Show progress by default for single worker, suppress for parallel
    verbose = (args.workers == 1) and not args.quiet

    print(f"Found {len(folds)} folds: {folds}")
    print(f"Running with {args.workers} workers...")
    if verbose:
        print("Verbose mode: showing real-time training output")
    else:
        print(f"Parallel mode: output logged to <output_dir>/fold_N/training.log")
        print(f"  To follow progress: tail -f {args.output_dir}/fold_*/training.log")

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare worker args
    worker_args = []
    for fold in folds:
        worker_args.append({
            'fold': fold,
            'config': args.config,
            'n_folds': args.n_folds,
            'output_dir': args.output_dir,
            'verbose': verbose
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
