#!/usr/bin/env python3
"""
Multi-Experiment MIL Training Script
Runs multiple combinations of MIL models and encoders, generating metrics CSV and confusion matrices

Usage:
    python run_mil_experiments.py --config experiments_config.json

    Or with CLI args:
    python run_mil_experiments.py \\
        --csv-path /path/to/labels.csv \\
        --features-base /path/to/features/ \\
        --experiments "abmil.base.uni_v2.none,dftd.base.uni_v2.none"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from tqdm import tqdm
import pandas as pd
import os
import argparse
from datetime import datetime
from pathlib import Path
import json

# Import shared utilities
from utils import preprocess_panda_data, create_dataloaders, plot_confusion_matrix
from src.builder import create_model

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_model(
    model_config: str,
    display_name: str,
    feats_path: str,
    df: pd.DataFrame,
    num_classes: int,
    class_labels: list,
    device: torch.device,
    output_dir: str,
    batch_size: int = 1,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    early_stopping_patience: int = 5,
    min_epochs: int = 10,
    feature_dropout_rate: float = 0.1,
    model_dropout_rate: float = 0.25,
    train_seed: int = 42,
):
    """
    Train and evaluate a single model configuration

    Args:
        model_config: Model configuration string (e.g., 'abmil.base.uni_v2.pc108-24k')
        display_name: Human-readable name for the experiment
        feats_path: Path to feature files
        df: Preprocessed dataframe
        num_classes: Number of output classes
        class_labels: List of class label names
        device: torch device
        output_dir: Directory to save results
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        early_stopping_patience: Patience for early stopping
        min_epochs: Minimum epochs before early stopping
        feature_dropout_rate: Dropout rate for features
        model_dropout_rate: Dropout rate for model
        train_seed: Random seed for training

    Returns:
        dict: Dictionary containing all metrics
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {display_name}")
    print(f"Model Config: {model_config}")
    print("="*80 + "\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        feats_path, df,
        batch_size=batch_size,
        num_workers=4,
        seed=train_seed
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}\n")

    # Create model
    print("Loading model...")
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    # Conditionally pass gate parameter (only for models that support it)
    model_kwargs = {
        'num_classes': num_classes,
        'dropout': model_dropout_rate,
    }
    # DFTD doesn't use the gate parameter, so only add it for other models
    if not model_config.lower().startswith('dftd'):
        model_kwargs['gate'] = True

    model = create_model(model_config, **model_kwargs).to(device)

    feature_dropout = nn.Dropout(p=feature_dropout_rate).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training setup
    print("Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Model save path
    model_save_name = display_name.replace(' ', '_').replace('+', '').lower()
    model_save_path = os.path.join(output_dir, f'best_model_{model_save_name}.pth')

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.

        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            features, labels = features.to(device), labels.to(device)


            features = feature_dropout(features)

            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                loss = results_dict['loss']

            # Scaled Backward Pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                features, labels = features.to(device), labels.to(device)

                features = feature_dropout(features)

                with torch.cuda.amp.autocast():
                    results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                    logits = results_dict['logits']
                    loss = results_dict['loss']

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)

        scheduler.step()

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  >>> Saved best model (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience and (epoch + 1) >= min_epochs:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

        print("-"*70)

    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")

    print("Loading best model for testing...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Testing'):
            features, labels = features.to(device), labels.to(device)

            features = F.normalize(features, p=2, dim=1)
            # NO dropout during testing!

            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                logits = results_dict['logits']

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"\nTest Results:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Balanced Acc:     {test_balanced_acc:.4f}")
    print(f"  Quadratic Kappa:  {test_kappa:.4f}\n")

    # Generate confusion matrix
    plot_labels = [label.replace(' ', '\n') for label in class_labels]
    cm_filename = f'confusion_matrix_{model_save_name}.png'
    cm_path = os.path.join(output_dir, cm_filename)

    plot_confusion_matrix(
        all_labels, all_preds, plot_labels,
        title=f'Confusion Matrix - {display_name}',
        output_path=cm_path
    )
    print(f"Confusion matrix saved to: {cm_path}")

    # Return metrics
    return {
        'experiment_name': display_name,
        'model_config': model_config,
        'test_accuracy': test_acc,
        'test_balanced_accuracy': test_balanced_acc,
        'test_quadratic_kappa': test_kappa,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'model_path': model_save_path,
        'confusion_matrix_path': cm_path,
    }

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Experiment MIL Training')
    parser.add_argument('--config', type=str, help='Path to experiments config JSON')
    parser.add_argument('--csv-path', type=str, help='Path to labels CSV')
    parser.add_argument('--features-base', type=str, help='Base path for features (encoder subdirs expected)')
    parser.add_argument('--feature-paths', type=str, help='JSON string mapping encoder->path, e.g. \'{"uni_v2":"/path/to/uni_v2/"}\'')
    parser.add_argument('--experiments', type=str, help='Comma-separated model configs, e.g. "abmil.base.uni_v2.none,dftd.base.uni_v2.none"')
    parser.add_argument('--output-dir', type=str, default='experiment_results', help='Output directory')
    parser.add_argument('--grade-group', action='store_true', help='Enable grade grouping')
    parser.add_argument('--exclude-mid-grade', action='store_true', help='Exclude mid-grade samples')
    parser.add_argument('--seed', type=int, default=10, help='Data split seed')
    parser.add_argument('--train-seed', type=int, default=42, help='Training seed')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--min-epochs', type=int, default=10, help='Minimum epochs before early stopping')
    parser.add_argument('--feature-dropout', type=float, default=0.1, help='Feature dropout rate')
    parser.add_argument('--model-dropout', type=float, default=0.25, help='Model dropout rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from file or build from args
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        csv_path = config['data_config']['csv_path']
        feature_paths = config.get('feature_paths', {})
        experiments = [(exp[0], exp[1]) for exp in config['experiments']]
        grade_group = config['data_config'].get('grade_group', False)
        exclude_mid_grade = config['data_config'].get('exclude_mid_grade', False)
        seed = config['data_config'].get('seed', 10)
        train_seed = config['data_config'].get('train_seed', 42)
        num_epochs = config['training_config'].get('num_epochs', 20)
        batch_size = config['training_config'].get('batch_size', 1)
        learning_rate = config['training_config'].get('learning_rate', 1e-4)
        weight_decay = config['training_config'].get('weight_decay', 1e-5)
        early_stopping_patience = config['training_config'].get('early_stopping_patience', 5)
        min_epochs = config['training_config'].get('min_epochs', 10)
        feature_dropout_rate = config['training_config'].get('feature_dropout_rate', 0.1)
        model_dropout_rate = config['training_config'].get('model_dropout_rate', 0.25)
        output_dir = config.get('output_dir', args.output_dir)
    else:
        if not args.csv_path:
            print("Error: --csv-path is required when not using --config")
            print("\nUsage:")
            print("  python run_mil_experiments.py --config experiments_config.json")
            print("  python run_mil_experiments.py --csv-path labels.csv --features-base /path/to/features/ --experiments 'abmil.base.uni_v2.none'")
            exit(1)

        csv_path = args.csv_path
        grade_group = args.grade_group
        exclude_mid_grade = args.exclude_mid_grade
        seed = args.seed
        train_seed = args.train_seed
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        early_stopping_patience = args.early_stopping_patience
        min_epochs = args.min_epochs
        feature_dropout_rate = args.feature_dropout
        model_dropout_rate = args.model_dropout
        output_dir = args.output_dir

        # Build feature paths
        if args.feature_paths:
            feature_paths = json.loads(args.feature_paths)
        elif args.features_base:
            # Auto-discover encoder directories
            features_base = Path(args.features_base)
            feature_paths = {}
            for subdir in features_base.iterdir():
                if subdir.is_dir() and subdir.name.startswith('features_'):
                    encoder = subdir.name.replace('features_', '')
                    feature_paths[encoder] = str(subdir)
            if not feature_paths:
                # Assume features_base is the direct path
                feature_paths = {'default': str(features_base)}
        else:
            print("Error: --features-base or --feature-paths is required when not using --config")
            exit(1)

        # Parse experiments
        if args.experiments:
            experiments = [(exp.strip(), exp.strip()) for exp in args.experiments.split(',')]
        else:
            print("Error: --experiments is required when not using --config")
            exit(1)

    print("="*80)
    print("MULTI-EXPERIMENT MIL TRAINING")
    print(f"Total experiments: {len(experiments)}")
    print("="*80 + "\n")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = os.path.join(output_dir, f'run_{timestamp}')
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {run_output_dir}\n")

    # Save experiment configuration
    config_save_path = os.path.join(run_output_dir, 'experiment_config.json')
    with open(config_save_path, 'w') as f:
        json.dump({
            'experiments': experiments,
            'feature_paths': feature_paths,
            'data_config': {
                'csv_path': csv_path,
                'grade_group': grade_group,
                'exclude_mid_grade': exclude_mid_grade,
                'seed': seed,
                'train_seed': train_seed,
            },
            'training_config': {
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'early_stopping_patience': early_stopping_patience,
                'min_epochs': min_epochs,
                'feature_dropout_rate': feature_dropout_rate,
                'model_dropout_rate': model_dropout_rate,
            }
        }, f, indent=2)
    print(f"Experiment configuration saved to: {config_save_path}\n")

    # Extract encoder from model config to get feature path
    def get_encoder_from_config(model_config: str) -> str:
        """Extract encoder name from model config string"""
        parts = model_config.split('.')
        if len(parts) >= 3:
            encoder = parts[2]  # e.g., 'uni_v2' from 'abmil.base.uni_v2.pc108-24k'
            return encoder
        return None

    # Get first encoder to preprocess data (assumes all use same data split)
    first_encoder = get_encoder_from_config(experiments[0][0])
    first_feats_path = feature_paths.get(first_encoder) or feature_paths.get('default')

    if not first_feats_path:
        raise ValueError(f"Feature path not found for encoder: {first_encoder}. Available: {list(feature_paths.keys())}")

    # Preprocess data once (data split will be the same for all experiments)
    print("="*70)
    print("PREPROCESSING DATA")
    print("="*70 + "\n")

    df, num_classes, class_labels = preprocess_panda_data(
        csv_path, first_feats_path,
        grade_group=grade_group,
        exclude_mid_grade=exclude_mid_grade,
        seed=seed
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Run experiments
    all_results = []

    for idx, (model_config, display_name) in enumerate(experiments, 1):
        print("\n" + "#"*80)
        print(f"RUNNING EXPERIMENT {idx}/{len(experiments)}")
        print("#"*80)

        # Get encoder and corresponding feature path
        encoder = get_encoder_from_config(model_config)
        feats_path = feature_paths.get(encoder) or feature_paths.get('default')

        if not feats_path:
            print(f"WARNING: Feature path not found for encoder '{encoder}'. Skipping...")
            continue

        if not os.path.exists(feats_path):
            print(f"WARNING: Feature path does not exist: {feats_path}. Skipping...")
            continue

        # Train and evaluate
        try:
            results = train_and_evaluate_model(
                model_config=model_config,
                display_name=display_name,
                feats_path=feats_path,
                df=df,
                num_classes=num_classes,
                class_labels=class_labels,
                device=device,
                output_dir=run_output_dir,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stopping_patience=early_stopping_patience,
                min_epochs=min_epochs,
                feature_dropout_rate=feature_dropout_rate,
                model_dropout_rate=model_dropout_rate,
                train_seed=train_seed,
            )
            all_results.append(results)

        except Exception as e:
            print(f"\nERROR in experiment '{display_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with next experiment...\n")
            continue

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_results_path = os.path.join(run_output_dir, 'experiment_results.csv')
        results_df.to_csv(csv_results_path, index=False)

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {csv_results_path}\n")
        print("Summary:")
        print(results_df[['experiment_name', 'test_accuracy', 'test_balanced_accuracy', 'test_quadratic_kappa']].to_string(index=False))
        print("\n")
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()
