#!/usr/bin/env python3
"""
Multi-Experiment MIL Training Script with PREDEFINED SPLITS
Uses k=all.tsv for predefined train/test splits instead of random splitting.
Optionally creates validation set from training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import json

# Import shared utilities
from utils import load_panda_predefined_splits, create_dataloaders, plot_confusion_matrix
from src.builder import create_model

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Define experiment combinations
EXPERIMENTS = [
    # Format: (model_config, display_name)
    ('abmil.base.uni_v2.pc108-24k', 'ABMIL Trained + UNI_v2'),
    ('dftd.base.uni_v2.none', 'DFTD Non-trained + UNI_v2'),
]

# Data paths - Update these to match your setup
TSV_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/k=all.tsv'  # Predefined splits
WSI_DIR = '/media/nadim/Data/prostate-cancer-grade-assessment/train_images'

# Feature paths for each encoder
FEATURE_PATHS = {
    'gigapath': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_gigapath/',
    'uni_v2': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/',
    'conch_v15': '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_conch_v15/',
}

# Data settings
FOLD_COLUMN = 'fold_0'  # Column in TSV containing train/test split
VAL_FRACTION = 0.1  # Create 10% validation set from training data (set to 0 for no validation)
GRADE_GROUP = False  # Use original ISUP grades (0-5) or group them
EXCLUDE_MID_GRADE = False  # Only applies if GRADE_GROUP=True
SEED = 10
TRAIN_SEED = 42

# Training settings per paper [cite: 849, 850, 852]
NUM_EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Early Stopping & Dropout
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS = 10
FEATURE_DROPOUT_RATE = 0.1  # [cite: 852]
MODEL_DROPOUT_RATE = 0.25   # [cite: 852]

# Output directory
OUTPUT_DIR = 'experiment_results_predefined'

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
    output_dir: str
):
    """
    Train and evaluate a single model configuration
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT: {display_name}")
    print(f"Model Config: {model_config}")
    print("="*80 + "\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        feats_path, df,
        batch_size=BATCH_SIZE,
        num_workers=4,
        seed=TRAIN_SEED
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}\n")

    # Create model
    print("Loading model...")
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)

    # Conditionally pass gate parameter (only for models that support it)
    model_kwargs = {
        'num_classes': num_classes,
        'dropout': MODEL_DROPOUT_RATE,
    }
    # DFTD doesn't use the gate parameter
    if not model_config.lower().startswith('dftd'):
        model_kwargs['gate'] = True

    model = create_model(model_config, **model_kwargs).to(device)

    # Dropout applied to input features [cite: 852]
    feature_dropout = nn.Dropout(p=FEATURE_DROPOUT_RATE).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Training setup [cite: 849]
    print("Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # Initialization for tracking best model
    best_val_loss = float('inf')
    best_val_kappa = -1.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Model save path
    model_save_name = display_name.replace(' ', '_').replace('+', '').lower()
    model_save_path = os.path.join(output_dir, f'best_model_{model_save_name}.pth')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0.

        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]'):
            features, labels = features.to(device), labels.to(device)
            features = feature_dropout(features)

            optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda'):
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
            for features, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]'):
                features, labels = features.to(device), labels.to(device)

                with torch.amp.autocast('cuda'):
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

        # Calculate Kappa for checkpointing
        val_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        scheduler.step()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  Val Kappa:  {val_kappa:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping & Checkpointing Logic
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            torch.save(model.state_dict(), model_save_path)
            print(f"  >>> Saved best model (Val Kappa: {best_val_kappa:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE and (epoch + 1) >= MIN_EPOCHS:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

        print("-"*70)

    # Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")

    print("Loading best model for testing...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    else:
        print("Warning: No saved model found. Using current weights.")

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Testing'):
            features, labels = features.to(device), labels.to(device)

            with torch.amp.autocast('cuda'):
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
        'best_val_kappa': best_val_kappa,
        'final_epoch': epoch + 1,
        'model_path': model_save_path,
        'confusion_matrix_path': cm_path,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("MULTI-EXPERIMENT MIL TRAINING (PREDEFINED SPLITS)")
    print(f"Total experiments: {len(EXPERIMENTS)}")
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
            'data_config': {
                'tsv_path': TSV_PATH,
                'fold_column': FOLD_COLUMN,
                'val_fraction': VAL_FRACTION,
                'grade_group': GRADE_GROUP,
                'exclude_mid_grade': EXCLUDE_MID_GRADE,
                'seed': SEED,
                'train_seed': TRAIN_SEED,
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
    print(f"Experiment configuration saved to: {config_save_path}\n")

    # Extract encoder from model config to get feature path
    def get_encoder_from_config(model_config: str) -> str:
        """Extract encoder name from model config string"""
        parts = model_config.split('.')
        if len(parts) >= 3:
            encoder = parts[2]  # e.g., 'uni_v2' from 'abmil.base.uni_v2.pc108-24k'
            return encoder
        return None

    # Get first encoder to load data (assumes all use same data split)
    first_encoder = get_encoder_from_config(EXPERIMENTS[0][0])
    first_feats_path = FEATURE_PATHS.get(first_encoder)

    if not first_feats_path:
        raise ValueError(f"Feature path not found for encoder: {first_encoder}")

    # Load data with predefined splits
    print("="*70)
    print("LOADING DATA WITH PREDEFINED SPLITS")
    print("="*70 + "\n")

    df, num_classes, class_labels = load_panda_predefined_splits(
        TSV_PATH, first_feats_path,
        fold_column=FOLD_COLUMN,
        val_fraction=VAL_FRACTION,
        grade_group=GRADE_GROUP,
        exclude_mid_grade=EXCLUDE_MID_GRADE,
        seed=SEED
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # Run experiments
    all_results = []

    for idx, (model_config, display_name) in enumerate(EXPERIMENTS, 1):
        print("\n" + "#"*80)
        print(f"RUNNING EXPERIMENT {idx}/{len(EXPERIMENTS)}")
        print("#"*80)

        # Get encoder and corresponding feature path
        encoder = get_encoder_from_config(model_config)
        feats_path = FEATURE_PATHS.get(encoder)

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
                output_dir=output_dir
            )
            all_results.append(results)

        except Exception as e:
            print(f"\nERROR in experiment '{display_name}': {str(e)}")
            # Print full traceback for debugging if needed
            import traceback
            traceback.print_exc()
            print("Continuing with next experiment...\n")
            continue

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'experiment_results.csv')
        results_df.to_csv(csv_path, index=False)

        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {csv_path}\n")
        print("Summary:")
        print(results_df[['experiment_name', 'test_accuracy', 'test_balanced_accuracy', 'test_quadratic_kappa']].to_string(index=False))
        print("\n")
    else:
        print("\nNo experiments completed successfully.")

if __name__ == "__main__":
    main()
