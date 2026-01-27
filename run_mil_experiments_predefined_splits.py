#!/usr/bin/env python3
"""
MIL Training Script using Predefined JSON Splits.
Supports K-fold cross-validation by running one fold at a time.
"""

import torch
import argparse
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from training.config import ExperimentConfig, DataConfig, TrainConfig, TaskType

from training.trainer import MILTrainer

from training.evaluator import evaluate, print_evaluation_results

from data_loading.dataset import MILDataset

from data_loading.pytorch_adapter import create_dataloader

from src.builder import create_model



def run_fold(config: ExperimentConfig, split_dir: str, fold: int):

    """Run a single fold of training."""

    print(f"\n{'='*80}")

    print(f"RUNNING FOLD {fold} from {split_dir}")

    print(f"{'='*80}\n")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    # 1. Initialize Base Dataset

    base_ds = MILDataset(config.data.labels_csv, config.data.features_dir)

    

    # 2. Apply multi-slide fusion if needed

    if config.data.hierarchical:

        print("Using Hierarchical fusion (Late Fusion)")

        full_ds = base_ds.group_by(config.data.group_column)

    else:

        print("Using Grouped fusion (Early Fusion)")

        full_ds = base_ds.concat_by(config.data.group_column)



    # 3. Load predefined split

    split_json = os.path.join(split_dir, f'splits_{fold}.json')

    print(f"Loading split from {split_json}...")

    

    train_ds = full_ds.load_split(split_json, 'train')

    val_ds = full_ds.load_split(split_json, 'val')

    test_ds = full_ds.load_split(split_json, 'test')



    print(f"Split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")



    # 4. Create Dataloaders

    train_loader, train_adapter = create_dataloader(

        train_ds, 

        batch_size=config.train.batch_size, 

        weighted_sampling=config.train.weighted_sampling,

        num_workers=config.data.num_workers,

        seed=config.train.seed

    )

    val_loader, _ = create_dataloader(

        val_ds, 

        batch_size=config.train.batch_size, 

        shuffle=False,

        num_workers=config.data.num_workers,

        label_map=train_adapter.label_map

    )

    test_loader, _ = create_dataloader(

        test_ds, 

        batch_size=1, 

        shuffle=False,

        num_workers=config.data.num_workers,

        label_map=train_adapter.label_map

    )



    # 6. Create Model
    model = create_model(
        config.model_name,
        num_classes=config.num_classes,
        dropout=config.train.model_dropout,
        num_heads=config.num_heads
    )

    # 7. Train
    fold_output_dir = Path(config.output_dir) / f"fold_{fold}"
    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
        device=device,
        checkpoint_dir=str(fold_output_dir)
    )

    history = trainer.fit() 
    
    # 8. Evaluate best model
    print("\nEvaluating best model on test set...")
    trainer.load_best_model()
    test_results = evaluate(
        trainer.model, 
        test_loader, 
        device, 
        use_amp=config.train.use_amp,
        task_type=config.train.task_type.value
    )
    
    print_evaluation_results(test_results)
    
    # Save results
    results_path = fold_output_dir / "test_results.json"
    # Convert numpy cm to list for JSON
    test_results_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in test_results.items() if k not in ['predictions', 'labels']}
    with open(results_path, 'w') as f:
        json.dump(test_results_serializable, f, indent=4)
    
    return test_results_serializable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MIL model with predefined splits')
    parser.add_argument('--config', type=str, required=True, help='Path to ExperimentConfig JSON')
    parser.add_argument('--split_dir', type=str, required=True, help='Directory containing splits_N.json')
    parser.add_argument('--fold', type=int, required=True, help='Fold index to run')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Load config
    config = ExperimentConfig.load(args.config)
    
    # Overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run fold
    run_fold(config, args.split_dir, args.fold)