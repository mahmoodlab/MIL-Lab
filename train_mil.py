#!/usr/bin/env python3
"""
MIL Training Entry Point

A clean, modular training script that uses the new data_loading and training modules.

Usage:
    python train_mil.py  # Uses default config
    python train_mil.py --config path/to/config.json

Example Config:
    {
        "data": {
            "labels_csv": "/path/to/labels.csv",
            "features_dir": "/path/to/features/",
            "split_column": "split"
        },
        "train": {
            "num_epochs": 20,
            "learning_rate": 1e-4
        },
        "model_name": "abmil.base.uni_v2.pc108-24k",
        "num_classes": 6,
        "output_dir": "experiments"
    }
"""

import torch
import argparse
from datetime import datetime
from pathlib import Path

from data_loading import MILDataset, create_dataloader
from training import ExperimentConfig, DataConfig, TrainConfig, MILTrainer, evaluate, print_evaluation_results
from src.builder import create_model
from utils import plot_confusion_matrix


def main(config: ExperimentConfig):
    """
    Main training function.

    Args:
        config: Experiment configuration
    """
    print("=" * 80)
    print("MIL TRAINING")
    print(f"Model: {config.model_name}")
    print(f"Output: {config.output_dir}")
    print("=" * 80 + "\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Set seeds
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.output_dir) / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}\n")

    # Save config
    config.save(run_dir / 'config.json')

    # Load dataset
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70 + "\n")

    dataset = MILDataset(
        labels_csv=config.data.labels_csv,
        features_dir=config.data.features_dir,
    )

    print(f"Total slides: {len(dataset)}")
    print(f"Embed dim: {dataset.embed_dim}")
    print(f"Num classes: {dataset.num_classes}\n")

    # Split dataset
    if config.data.split_column:
        print(f"Using predefined splits from column: {config.data.split_column}")
        splits = dataset.split_by_column(config.data.split_column)
    else:
        print(f"Using random splits (train={config.data.train_frac}, val={config.data.val_frac})")
        splits = dataset.random_split(
            train_frac=config.data.train_frac,
            val_frac=config.data.val_frac,
            seed=config.data.seed,
        )

    print("\nSplit sizes:")
    for name, split_dataset in splits.items():
        print(f"  {name}: {len(split_dataset)} slides")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, train_adapter = create_dataloader(
        splits['train'],
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        weighted_sampling=config.train.weighted_sampling,
        seed=config.train.seed,
    )

    # Use same label_map for val and test
    label_map = train_adapter.label_map

    val_loader, _ = create_dataloader(
        splits['val'],
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        weighted_sampling=False,
        label_map=label_map,
    )

    test_loader, _ = create_dataloader(
        splits['test'],
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        weighted_sampling=False,
        label_map=label_map,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70 + "\n")

    model_kwargs = {
        'num_classes': config.num_classes,
        'dropout': config.train.model_dropout,
    }
    # DFTD doesn't use the gate parameter
    if not config.model_name.lower().startswith('dftd'):
        model_kwargs['gate'] = True

    model = create_model(config.model_name, **model_kwargs).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70 + "\n")

    trainer = MILTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.train,
        device=device,
        checkpoint_dir=run_dir,
    )

    history = trainer.fit()

    # Load best model for evaluation
    trainer.load_best_model()

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70 + "\n")

    results = evaluate(model, test_loader, device, use_amp=config.train.use_amp)

    # Get class labels for display
    inverse_label_map = {v: k for k, v in label_map.items()}
    class_labels = [inverse_label_map[i] for i in range(len(label_map))]

    print_evaluation_results(results, class_labels)

    # Save confusion matrix plot
    cm_path = run_dir / 'confusion_matrix.png'
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_labels,
        title=f'Confusion Matrix - {config.experiment_name}',
        output_path=str(cm_path),
    )
    print(f"Confusion matrix saved to: {cm_path}")

    # Save results summary
    summary = {
        'model_name': config.model_name,
        'test_accuracy': results['accuracy'],
        'test_balanced_accuracy': results['balanced_accuracy'],
        'test_quadratic_kappa': results['quadratic_kappa'],
        'best_val_kappa': trainer.best_val_kappa,
        'best_epoch': trainer.best_epoch + 1,
        'total_epochs': len(history['train_loss']),
    }

    import json
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {run_dir / 'results.json'}")

    # Print final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest Validation Kappa: {trainer.best_val_kappa:.4f} (epoch {trainer.best_epoch + 1})")
    print(f"Test Accuracy:         {results['accuracy']:.4f}")
    print(f"Test Balanced Acc:     {results['balanced_accuracy']:.4f}")
    print(f"Test Quadratic Kappa:  {results['quadratic_kappa']:.4f}")
    print(f"\nAll outputs saved to: {run_dir}")

    return results, history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MIL Training Script')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config JSON file',
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='Path to labels CSV file',
    )
    parser.add_argument(
        '--features-dir',
        type=str,
        default=None,
        help='Path to features directory',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='abmil.base.uni_v2.pc108-24k',
        help='Model name (default: abmil.base.uni_v2.pc108-24k)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=6,
        help='Number of classes (default: 6)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs (default: 20)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Output directory (default: experiments)',
    )
    parser.add_argument(
        '--split-column',
        type=str,
        default=None,
        help='Column name for predefined splits',
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=100,
        help='Early stopping patience (default: 100)',
    )
    parser.add_argument(
        '--min-epochs',
        type=int,
        default=10,
        help='Minimum epochs before early stopping (default: 10)',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.config:
        # Load from config file
        config = ExperimentConfig.load(args.config)
    else:
        # Build config from args
        if not args.labels_csv or not args.features_dir:
            print("Error: --labels-csv and --features-dir are required when not using --config")
            print("\nExample usage:")
            print("  python train_mil.py --labels-csv labels.csv --features-dir /path/to/features")
            print("  python train_mil.py --config config.json")
            exit(1)

        config = ExperimentConfig(
            data=DataConfig(
                labels_csv=args.labels_csv,
                features_dir=args.features_dir,
                split_column=args.split_column,
            ),
            train=TrainConfig(
                num_epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                min_epochs=args.min_epochs,
            ),
            model_name=args.model,
            num_classes=args.num_classes,
            output_dir=args.output_dir,
        )

    main(config)
