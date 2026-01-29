#!/usr/bin/env python3
"""
MIL Training Script with K-Fold Cross-Validation.

Uses ExperimentConfig for all settings and supports:
- Binary/multiclass classification via task_type
- Case-level grouping with early/late fusion
- Stratified K-fold CV with proper data leakage prevention
- Optional single-fold execution for parallel orchestration

Usage:
    # Run all folds sequentially
    python run_mil_experiments_cv.py --config experiment.json --n_folds 5

    # Run a specific fold (for parallel execution via orchestrator)
    python run_mil_experiments_cv.py --config experiment.json --n_folds 5 --fold 0

    # Override output directory
    python run_mil_experiments_cv.py --config experiment.json --output_dir results/
"""

import torch
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from training.config import ExperimentConfig, DataConfig, TrainConfig, TaskType
from training.trainer import MILTrainer
from training.evaluator import evaluate, print_evaluation_results
from data_loading.dataset import MILDataset
from data_loading.pytorch_adapter import create_dataloader
from src.builder import create_model


def run_fold(
    config: ExperimentConfig,
    full_ds,
    fold: int,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    n_folds: int,
):
    """
    Run a single fold of cross-validation.

    Args:
        config: Experiment configuration
        full_ds: Full dataset (GroupedMILDataset or HierarchicalMILDataset)
        fold: Current fold number (0-indexed)
        train_ids: List of group IDs for training
        val_ids: List of group IDs for validation
        test_ids: List of group IDs for testing
        n_folds: Total number of folds

    Returns:
        dict: Test results for this fold
    """
    print(f"\n{'='*80}")
    print(f"RUNNING FOLD {fold + 1}/{n_folds}")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create subsets
    train_ds = full_ds.get_subset(train_ids)
    val_ds = full_ds.get_subset(val_ids)
    test_ds = full_ds.get_subset(test_ids)

    print(f"Split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Create Dataloaders
    train_loader, train_adapter = create_dataloader(
        train_ds,
        batch_size=config.train.batch_size,
        weighted_sampling=config.train.weighted_sampling,
        num_workers=config.data.num_workers,
        seed=config.train.seed + fold,  # Different seed per fold
    )
    val_loader, _ = create_dataloader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        label_map=train_adapter.label_map,
    )
    test_loader, _ = create_dataloader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        label_map=train_adapter.label_map,
    )

    # Create Model (with different seed per fold for initialization)
    torch.manual_seed(config.train.seed + fold)
    torch.cuda.manual_seed_all(config.train.seed + fold)

    model = create_model(
        config.model_name,
        num_classes=config.num_classes,
        dropout=config.train.model_dropout,
        num_heads=config.num_heads,
    )

    # Train
    fold_output_dir = Path(config.output_dir) / f"fold_{fold}"
    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
        device=device,
        checkpoint_dir=str(fold_output_dir),
    )

    history = trainer.fit()

    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    trainer.load_best_model()
    test_results = evaluate(
        trainer.model,
        test_loader,
        device,
        use_amp=config.train.use_amp,
        task_type=config.train.task_type.value,
    )

    print_evaluation_results(test_results)

    # Save results
    results_path = fold_output_dir / "test_results.json"
    test_results_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in test_results.items()
        if k not in ['predictions', 'labels', 'probabilities']
    }
    test_results_serializable['fold'] = fold

    with open(results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=4)

    print(f"Results saved to: {results_path}")

    return test_results_serializable


def generate_cv_splits(full_ds, n_folds: int, seed: int, val_frac: float = 0.1):
    """
    Generate stratified K-fold splits.

    Args:
        full_ds: Dataset with group_ids and labels properties
        n_folds: Number of folds
        seed: Random seed
        val_frac: Fraction of train+val to use for validation

    Yields:
        tuple: (fold_idx, train_ids, val_ids, test_ids)
    """
    from sklearn.model_selection import train_test_split

    group_ids = np.array(full_ds.group_ids)
    labels = np.array(full_ds.labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(group_ids, labels)):
        train_val_ids = group_ids[train_val_idx]
        test_ids = group_ids[test_idx].tolist()

        # Further split train_val into train and val
        train_val_labels = labels[train_val_idx]
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=val_frac,
            stratify=train_val_labels,
            random_state=seed,
        )

        yield fold_idx, train_ids.tolist(), val_ids.tolist(), test_ids


def aggregate_results(output_dir: str, n_folds: int):
    """Aggregate results from all folds and print summary."""
    all_results = []

    for fold in range(n_folds):
        results_path = Path(output_dir) / f"fold_{fold}" / "test_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No results found to aggregate.")
        return

    # Extract numeric metrics
    metrics = {}
    for key in all_results[0].keys():
        if key in ['fold', 'confusion_matrix']:
            continue
        values = [r.get(key) for r in all_results if r.get(key) is not None]
        if values and isinstance(values[0], (int, float)):
            metrics[key] = values

    # Print summary
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    # Save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(output_dir) / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nDetailed results saved to: {summary_path}")

    # Save aggregated stats
    stats = {metric: {'mean': np.mean(values), 'std': np.std(values)}
             for metric, values in metrics.items()}
    stats_path = Path(output_dir) / "cv_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Aggregated stats saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='MIL Training with K-Fold Cross-Validation')
    parser.add_argument('--config', type=str, required=True, help='Path to ExperimentConfig JSON')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=None, help='Run only this fold (0-indexed). If not set, runs all folds.')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of train set to use for validation')

    args = parser.parse_args()

    # Load config
    config = ExperimentConfig.load(args.config)

    # Override output_dir if specified
    if args.output_dir:
        config.output_dir = args.output_dir

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MIL TRAINING WITH K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Num classes: {config.num_classes}")
    print(f"Task type: {config.train.task_type.value}")
    print(f"Folds: {args.n_folds}")
    print(f"Output: {config.output_dir}")
    print("=" * 80 + "\n")

    # Save config copy to output
    config_save_path = Path(config.output_dir) / "experiment_config.json"
    config.save(str(config_save_path))

    # Initialize Base Dataset
    print("Loading dataset...")
    base_ds = MILDataset(config.data.labels_csv, config.data.features_dir)

    # Apply fusion strategy
    if config.data.hierarchical:
        print(f"Using Hierarchical fusion (Late Fusion) by '{config.data.group_column}'")
        full_ds = base_ds.group_by(config.data.group_column)
    else:
        print(f"Using Grouped fusion (Early Fusion) by '{config.data.group_column}'")
        full_ds = base_ds.concat_by(config.data.group_column)

    print(f"Total samples (after grouping): {len(full_ds)}")
    print(f"Number of classes: {full_ds.num_classes}")

    # Generate all splits upfront (ensures determinism)
    all_splits = list(generate_cv_splits(
        full_ds,
        n_folds=args.n_folds,
        seed=config.data.seed,
        val_frac=args.val_frac,
    ))

    # Save splits for reproducibility
    splits_save_path = Path(config.output_dir) / "cv_splits.json"
    splits_data = {
        f"fold_{i}": {"train": train, "val": val, "test": test}
        for i, train, val, test in all_splits
    }
    with open(splits_save_path, 'w') as f:
        json.dump(splits_data, f, indent=2)
    print(f"Splits saved to: {splits_save_path}")

    # Run folds
    if args.fold is not None:
        # Single fold mode (for parallel orchestration)
        if args.fold < 0 or args.fold >= args.n_folds:
            raise ValueError(f"Fold {args.fold} out of range [0, {args.n_folds})")

        fold_idx, train_ids, val_ids, test_ids = all_splits[args.fold]
        run_fold(config, full_ds, fold_idx, train_ids, val_ids, test_ids, args.n_folds)
    else:
        # Run all folds sequentially
        for fold_idx, train_ids, val_ids, test_ids in all_splits:
            run_fold(config, full_ds, fold_idx, train_ids, val_ids, test_ids, args.n_folds)

        # Aggregate results
        aggregate_results(config.output_dir, args.n_folds)

    print("\nDone!")


if __name__ == '__main__':
    main()
