#!/usr/bin/env python3
"""
Simplified ABMIL training script using shared utilities
(Final Corrected Version: Matches Paper 2506.09022v2 + UNI Feature Norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # <<< FIX: Added for normalization
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from tqdm import tqdm

# Import shared utilities
from utils import preprocess_panda_data, create_dataloaders, plot_confusion_matrix
from src.builder import create_model

# ============================================================================
# CONFIGURATION
# ============================================================================
CSV_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
FEATS_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/'
WSI_DIR = '/media/nadim/Data/prostate-cancer-grade-assessment/train_images'

# Data settings
GRADE_GROUP = True
EXCLUDE_MID_GRADE = True
SEED = 10
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

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Step 1: Preprocess data
    df, num_classes, class_labels = preprocess_panda_data(
        CSV_PATH, FEATS_PATH,
        grade_group=GRADE_GROUP,
        exclude_mid_grade=EXCLUDE_MID_GRADE,
        seed=SEED
    )

    # Step 2: Create dataloaders
    print("="*70)
    print("CREATING DATALOADERS")
    print("="*70 + "\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        FEATS_PATH, df,
        batch_size=BATCH_SIZE,
        num_workers=4,
        seed=TRAIN_SEED
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}\n")

    # Step 3: Create model
    print("="*70)
    print("LOADING ABMIL MODEL (PC-108 Pretrained)")
    print("="*70 + "\n")

    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        'abmil.base.uni_v2.pc108-24k',
        num_classes=num_classes,
        dropout=MODEL_DROPOUT_RATE,
        gate=True
    ).to(device)

    feature_dropout = nn.Dropout(p=FEATURE_DROPOUT_RATE).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 4: Training
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0 
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0.

        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]'):
            features, labels = features.to(device), labels.to(device)
            
            # <<< FIX 1: L2 Normalize UNI features (Standard practice for these encoders)
            features = F.normalize(features, p=2, dim=1)
            
            # Apply dropout
            features = feature_dropout(features)

            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                loss = results_dict['loss']

            # Scaled Backward Pass
            scaler.scale(loss).backward()
            
            # <<< FIX 2: Gradient Clipping (Crucial for Batch Size 1 stability)
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
                
                # Normalize features in Validation too
                features = F.normalize(features, p=2, dim=1)
                features = feature_dropout(features)

                # <<< FIX 3: Use Autocast in Validation for consistency
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
        
        # Step Cosine Scheduler
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_abmil.pth')
            print(f"  >>> Saved best model (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0 
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE and (epoch + 1) >= MIN_EPOCHS:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break
        
        print("-"*70)

    # Step 5: Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")

    print("Loading best model for testing...")
    model.load_state_dict(torch.load('best_model_abmil.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Testing'):
            features, labels = features.to(device), labels.to(device)
            
            # Consistency in Testing
            features = F.normalize(features, p=2, dim=1)
            features = feature_dropout(features)
            
            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                logits = results_dict['logits']
                
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"\nTest Results:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Balanced Acc:     {test_balanced_acc:.4f}")
    print(f"  Quadratic Kappa:  {test_kappa:.4f}")

    plot_labels = [label.replace(' ', '\n') for label in class_labels]
    plot_confusion_matrix(
        all_labels, all_preds, plot_labels,
        title='Confusion Matrix - ABMIL (Paper Replication)',
        output_path='confusion_matrix_abmil_replicated.png'
    )

    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()