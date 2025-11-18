#!/usr/bin/env python3
"""
Multi-Experiment MIL Training Script with K-Fold Cross-Validation
Runs multiple combinations of MIL models and encoders with cross-validation
Generates metrics CSV with mean ± std and confusion matrices for each fold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import json

# Import shared utilities
from utils import preprocess_panda_data, PANDAH5Dataset, plot_confusion_matrix
from src.builder import create_model

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Define experiment combinations
EXPERIMENTS = [
    # Format: (model_config, display_name)
    ('abmil.base.gigapath.pc108-24k', 'ABMIL + GigaPath'),
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL + UNI_v2'),
    ('abmil.base.conch_v15.pc108-24k', 'ABMIL + CONCH_v1.5'),
    # Add more combinations as needed:
    # ('transmil.base.gigapath.pc108-24k', 'TransMIL + GigaPath'),
    # ('transmil.base.uni_v2.pc108-24k', 'TransMIL + UNI_v2'),
    # ('clam.base.uni_v2.pc108-24k', 'CLAM + UNI_v2'),
]

# Data paths - Update these to match your setup
CSV_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
WSI_DIR = '/media/nadim/Data/prostate-cancer-grade-assessment/train_images'

# Feature paths for each encoder
FEATURE_PATHS = {
    'gigapath': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_gigapath/',
    'uni_v2': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/',
    'conch_v15': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_conch_v15/',
    # Add more encoders as needed
}

# Data settings
GRADE_GROUP = True
EXCLUDE_MID_GRADE = True
SEED = 10

# Cross-validation settings
N_FOLDS = 5  # Number of CV folds
TRAIN_SEED = 42

# Training settings
NUM_EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early Stopping & Dropout
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS = 10
FEATURE_DROPOUT_RATE = 0.1
MODEL_DROPOUT_RATE = 0.25

# Output directory
OUTPUT_DIR = 'experiment_results_cv'

# ============================================================================
# TRAINING FUNCTION FOR SINGLE FOLD
# ============================================================================

def train_and_evaluate_fold(
    model_config: str,
    display_name: str,
    feats_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_classes: int,
    class_labels: list,
    device: torch.device,
    fold_num: int,
    output_dir: str
):
    """
    Train and evaluate a single fold

    Args:
        model_config: Model configuration string
        display_name: Human-readable name for the experiment
        feats_path: Path to feature files
        train_df, val_df, test_df: DataFrames for each split
        num_classes: Number of output classes
        class_labels: List of class label names
        device: torch device
        fold_num: Current fold number
        output_dir: Directory to save results

    Returns:
        dict: Dictionary containing all metrics for this fold
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num}/{N_FOLDS}")
    print(f"{'='*80}\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataset = PANDAH5Dataset(feats_path, train_df, "train", num_features=None, seed=TRAIN_SEED)
    val_dataset = PANDAH5Dataset(feats_path, val_df, "val", num_features=None, seed=TRAIN_SEED)
    test_dataset = PANDAH5Dataset(feats_path, test_df, "test", num_features=None, seed=TRAIN_SEED)

    # Weighted sampling for training
    from torch.utils.data import WeightedRandomSampler
    train_labels = train_df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]

    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}\n")

    # Create model
    print("Loading model...")
    torch.manual_seed(TRAIN_SEED + fold_num)  # Different seed per fold
    torch.cuda.manual_seed_all(TRAIN_SEED + fold_num)

    model = create_model(
        model_config,
        num_classes=num_classes,
        dropout=MODEL_DROPOUT_RATE,
        gate=True
    ).to(device)

    feature_dropout = nn.Dropout(p=FEATURE_DROPOUT_RATE).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    # Model save path
    model_save_name = display_name.replace(' ', '_').replace('+', '').lower()
    model_save_path = os.path.join(output_dir, f'best_model_{model_save_name}_fold{fold_num}.pth')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0.

        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', leave=False):
            features, labels = features.to(device), labels.to(device)

            # L2 Normalize features
            features = F.normalize(features, p=2, dim=1)
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

        # Validation
        model.eval()
        val_loss = 0.
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)

                features = F.normalize(features, p=2, dim=1)
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
        val_acc = accuracy_score(all_labels, all_preds)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:  # Print every 5 epochs
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE and (epoch + 1) >= MIN_EPOCHS:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            features = F.normalize(features, p=2, dim=1)
            features = feature_dropout(features)

            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features)
                logits = results_dict['logits']

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_acc = accuracy_score(all_labels, all_preds)
    test_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"Fold {fold_num} Test Results:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Balanced Acc:     {test_balanced_acc:.4f}")
    print(f"  Quadratic Kappa:  {test_kappa:.4f}")

    # Generate confusion matrix
    plot_labels = [label.replace(' ', '\n') for label in class_labels]
    cm_filename = f'confusion_matrix_{model_save_name}_fold{fold_num}.png'
    cm_path = os.path.join(output_dir, cm_filename)

    plot_confusion_matrix(
        all_labels, all_preds, plot_labels,
        title=f'Confusion Matrix - {display_name} (Fold {fold_num})',
        output_path=cm_path
    )

    # Return metrics
    return {
        'fold': fold_num,
        'test_accuracy': test_acc,
        'test_balanced_accuracy': test_balanced_acc,
        'test_quadratic_kappa': test_kappa,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'model_path': model_save_path,
        'confusion_matrix_path': cm_path,
    }

# ============================================================================
# EXPERIMENT RUNNER WITH K-FOLD CV
# ============================================================================

def run_cv_experiment(
    model_config: str,
    display_name: str,
    feats_path: str,
    df: pd.DataFrame,
    num_classes: int,
    class_labels: list,
    device: torch.device,
    output_dir: str
):
    """
    Run k-fold cross-validation for a single experiment

    Args:
        model_config: Model configuration string
        display_name: Human-readable name
        feats_path: Path to features
        df: Full preprocessed dataframe
        num_classes: Number of classes
        class_labels: Class label names
        device: torch device
        output_dir: Output directory

    Returns:
        dict: Aggregated results across all folds
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {display_name}")
    print(f"Model Config: {model_config}")
    print(f"Running {N_FOLDS}-Fold Cross-Validation")
    print("="*80)

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Store results for each fold
    fold_results = []

    # Perform k-fold cross-validation
    for fold_num, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label']), 1):
        # Split data
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Further split train_val into train and val (90% train, 10% val)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.1, stratify=train_val_df['label'], random_state=SEED
        )

        # Add split column for compatibility with PANDAH5Dataset
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        # Train and evaluate this fold
        fold_result = train_and_evaluate_fold(
            model_config=model_config,
            display_name=display_name,
            feats_path=feats_path,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            num_classes=num_classes,
            class_labels=class_labels,
            device=device,
            fold_num=fold_num,
            output_dir=output_dir
        )

        fold_results.append(fold_result)

    # Aggregate results
    metrics = ['test_accuracy', 'test_balanced_accuracy', 'test_quadratic_kappa', 'best_val_loss']
    aggregated_results = {
        'experiment_name': display_name,
        'model_config': model_config,
        'n_folds': N_FOLDS,
    }

    # Calculate mean and std for each metric
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        aggregated_results[f'{metric}_mean'] = np.mean(values)
        aggregated_results[f'{metric}_std'] = np.std(values)

    # Store per-fold results
    aggregated_results['fold_results'] = fold_results

    # Print summary
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION SUMMARY - {display_name}")
    print("="*80)
    for metric in metrics:
        mean_val = aggregated_results[f'{metric}_mean']
        std_val = aggregated_results[f'{metric}_std']
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*80 + "\n")

    return aggregated_results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("MULTI-EXPERIMENT MIL TRAINING WITH K-FOLD CROSS-VALIDATION")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Folds per experiment: {N_FOLDS}")
    print("="*80 + "\n")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Save experiment configuration
    config_save_path = os.path.join(output_dir, 'experiment_config.json')
    with open(config_save_path, 'w') as f:
        json.dump({
            'experiments': [(config, name) for config, name in EXPERIMENTS],
            'cv_config': {
                'n_folds': N_FOLDS,
                'seed': SEED,
            },
            'data_config': {
                'csv_path': CSV_PATH,
                'grade_group': GRADE_GROUP,
                'exclude_mid_grade': EXCLUDE_MID_GRADE,
            },
            'training_config': {
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                'min_epochs': MIN_EPOCHS,
                'feature_dropout_rate': FEATURE_DROPOUT_RATE,
                'model_dropout_rate': MODEL_DROPOUT_RATE,
            }
        }, f, indent=2)
    print(f"Configuration saved to: {config_save_path}\n")

    # Extract encoder from model config
    def get_encoder_from_config(model_config: str) -> str:
        parts = model_config.split('.')
        if len(parts) >= 3:
            return parts[2]
        return None

    # Get first encoder to preprocess data
    first_encoder = get_encoder_from_config(EXPERIMENTS[0][0])
    first_feats_path = FEATURE_PATHS.get(first_encoder)

    if not first_feats_path:
        raise ValueError(f"Feature path not found for encoder: {first_encoder}")

    # Preprocess data once (without splitting - CV will handle splits)
    print("="*70)
    print("PREPROCESSING DATA")
    print("="*70 + "\n")

    # We'll use a custom preprocessing that doesn't split
    df_labels = pd.read_csv(CSV_PATH)[['slide_id', 'label']]
    df_labels['isup_grade'] = df_labels['label']

    # Apply grade grouping
    if GRADE_GROUP:
        if EXCLUDE_MID_GRADE:
            df_labels = df_labels[~df_labels['isup_grade'].isin([2, 3])].reset_index(drop=True)

        def map_isup_to_group(isup_grade):
            if isup_grade == 0:
                return 0
            elif isup_grade == 1:
                return 1
            elif isup_grade in [2, 3]:
                return 2
            elif isup_grade in [4, 5]:
                return 2 if EXCLUDE_MID_GRADE else 3
            else:
                raise ValueError(f"Invalid ISUP grade: {isup_grade}")

        df_labels['label'] = df_labels['isup_grade'].apply(map_isup_to_group)

    # Find available features
    from glob import glob
    feature_files = glob(os.path.join(first_feats_path, '*.h5'))
    available_slide_ids = [os.path.basename(f).replace('.h5', '') for f in feature_files]

    # Match with available features
    df_labels['has_features'] = df_labels['slide_id'].isin(available_slide_ids)
    df = df_labels[df_labels['has_features']].drop(columns=['has_features']).reset_index(drop=True)

    num_classes = len(df['label'].unique())

    # Define group names
    if GRADE_GROUP:
        if EXCLUDE_MID_GRADE:
            class_labels = ['Group 0 (No cancer)', 'Group 1 (Low grade)', 'Group 2 (High grade)']
        else:
            class_labels = ['Group 0 (No cancer)', 'Group 1 (Low grade)',
                           'Group 2 (Mid grade)', 'Group 3 (High grade)']
    else:
        class_labels = [f'ISUP {i}' for i in range(6)]

    print(f"Total slides: {len(df)}")
    print(f"Number of classes: {num_classes}\n")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Run experiments
    all_results = []

    for idx, (model_config, display_name) in enumerate(EXPERIMENTS, 1):
        print("\n" + "#"*80)
        print(f"RUNNING EXPERIMENT {idx}/{len(EXPERIMENTS)}")
        print("#"*80)

        # Get encoder and feature path
        encoder = get_encoder_from_config(model_config)
        feats_path = FEATURE_PATHS.get(encoder)

        if not feats_path:
            print(f"WARNING: Feature path not found for encoder '{encoder}'. Skipping...")
            continue

        if not os.path.exists(feats_path):
            print(f"WARNING: Feature path does not exist: {feats_path}. Skipping...")
            continue

        # Run CV experiment
        try:
            results = run_cv_experiment(
                model_config=model_config,
                display_name=display_name,
                feats_path=feats_path,
                df=df,
                num_classes=num_classes,
                class_labels=class_labels,
                device=device,
                output_dir=output_dir
            )
            all_results.append(results)

        except Exception as e:
            print(f"\nERROR in experiment '{display_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with next experiment...\n")
            continue

    # Save results
    if all_results:
        # Create summary CSV
        summary_rows = []
        detailed_rows = []

        for result in all_results:
            # Summary row (mean ± std)
            summary_row = {
                'experiment_name': result['experiment_name'],
                'model_config': result['model_config'],
                'n_folds': result['n_folds'],
                'test_accuracy_mean': result['test_accuracy_mean'],
                'test_accuracy_std': result['test_accuracy_std'],
                'test_balanced_accuracy_mean': result['test_balanced_accuracy_mean'],
                'test_balanced_accuracy_std': result['test_balanced_accuracy_std'],
                'test_quadratic_kappa_mean': result['test_quadratic_kappa_mean'],
                'test_quadratic_kappa_std': result['test_quadratic_kappa_std'],
                'best_val_loss_mean': result['best_val_loss_mean'],
                'best_val_loss_std': result['best_val_loss_std'],
            }
            summary_rows.append(summary_row)

            # Detailed rows (per fold)
            for fold_result in result['fold_results']:
                detailed_row = {
                    'experiment_name': result['experiment_name'],
                    'model_config': result['model_config'],
                    'fold': fold_result['fold'],
                    'test_accuracy': fold_result['test_accuracy'],
                    'test_balanced_accuracy': fold_result['test_balanced_accuracy'],
                    'test_quadratic_kappa': fold_result['test_quadratic_kappa'],
                    'best_val_loss': fold_result['best_val_loss'],
                    'final_epoch': fold_result['final_epoch'],
                    'model_path': fold_result['model_path'],
                    'confusion_matrix_path': fold_result['confusion_matrix_path'],
                }
                detailed_rows.append(detailed_row)

        # Save summary CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(output_dir, 'cv_summary_results.csv')
        summary_df.to_csv(summary_csv_path, index=False)

        # Save detailed CSV
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv_path = os.path.join(output_dir, 'cv_detailed_results.csv')
        detailed_df.to_csv(detailed_csv_path, index=False)

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"\nSummary results saved to: {summary_csv_path}")
        print(f"Detailed results saved to: {detailed_csv_path}\n")
        print("Summary (Mean ± Std):")
        print(summary_df[['experiment_name', 'test_accuracy_mean', 'test_accuracy_std',
                          'test_balanced_accuracy_mean', 'test_balanced_accuracy_std',
                          'test_quadratic_kappa_mean', 'test_quadratic_kappa_std']].to_string(index=False))
        print("\n")
    else:
        print("\nNo experiments completed successfully.")

if __name__ == "__main__":
    main()
