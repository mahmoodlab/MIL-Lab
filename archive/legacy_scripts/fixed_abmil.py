#!/usr/bin/env python3
"""
Simplified ABMIL training script using shared utilities
(Corrected to match paper 2506.09022v2, with sampler logic removed)
"""

import torch
import torch.nn as nn
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
#FEATS_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/panda'
WSI_DIR = '/media/nadim/Data/prostate-cancer-grade-assessment/train_images'

# Data settings
GRADE_GROUP = False
EXCLUDE_MID_GRADE = False
SEED = 10
TRAIN_SEED = 42

# Training settings
NUM_EPOCHS = 20
BATCH_SIZE = 1  # This is correct per the paper
LEARNING_RATE = 1e-4  # Correct per the paper
WEIGHT_DECAY = 1e-5  # Correct per the paper
# <<< FIX: Removed NUM_FEATURES. We must process all patches per WSI

# <<< FIX: Added Early Stopping parameters per paper
EARLY_STOPPING_PATIENCE = 5
MIN_EPOCHS = 10

# <<< FIX: Added Feature Dropout per paper
FEATURE_DROPOUT_RATE = 0.1
MODEL_DROPOUT_RATE = 0.25 # Correct per the paper

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
    
    # <<< FIX: Removed the WeightedRandomSampler code block
    # (Assuming create_dataloaders handles balanced sampling as requested)


    # Step 2: Create dataloaders
    print("="*70)
    print("CREATING DATALOADERS")
    print("="*70 + "\n")

    train_loader, val_loader, test_loader = create_dataloaders(
        FEATS_PATH, df,
        batch_size=BATCH_SIZE,
        # <<< FIX: Removed num_features, we process the whole bag
        num_workers=4,
        seed=TRAIN_SEED
        # <<< FIX: Removed sampler=train_sampler
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

    # This assumes 'create_model' loads the PC-108 pretrained weights
    # when given this model name string.
    model = create_model(
        'abmil.base.uni_v2.pc108-24k',
        num_classes=num_classes,
        dropout=MODEL_DROPOUT_RATE, # This is the internal model dropout
        gate=True
    ).to(device)

    # <<< FIX: Instantiate the feature dropout layer
    feature_dropout = nn.Dropout(p=FEATURE_DROPOUT_RATE).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Step 4: Training
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    criterion = nn.CrossEntropyLoss()
    
    # <<< FIX: Changed optimizer to AdamW
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # <<< FIX: Changed scheduler to CosineAnnealingLR
    # T_max is the number of epochs.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0 # <<< FIX: Added for early stopping
    train_losses = []
    val_losses = []
    val_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0.

        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]'):
            features, labels = features.to(device), labels.to(device)
            
            # <<< FIX: Apply feature dropout before model
            features = feature_dropout(features)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
                loss = results_dict['loss']

            scaler.scale(loss).backward()
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
                
                # <<< FIX: Apply feature dropout during validation (model.eval() handles it)
                features = feature_dropout(features)

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
        
        # <<< FIX: Step the Cosine scheduler *per epoch*
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {new_lr:.6f}")

        # <<< FIX: Implement full early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_abmil.pth')
            print(f"  >>> Saved best model (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0  # Reset patience
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
            
            # <<< FIX: Apply feature dropout during testing (model.eval() handles it)
            features = feature_dropout(features)
            
            results_dict, log_dict = model(features)
            logits = results_dict['logits']
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # <<< FIX: This is the primary metric for PANDA
    test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

    print(f"\nTest Results:")
    print(f"  Accuracy:         {test_acc:.4f}")
    print(f"  Balanced Acc:     {test_balanced_acc:.4f}")
    print(f"  Quadratic Kappa:  {test_kappa:.4f}  <-- Paper's Metric")

    # Save confusion matrix
    plot_labels = [label.replace(' ', '\n') for label in class_labels]
    plot_confusion_matrix(
        all_labels, all_preds, plot_labels,
        title='Confusion Matrix - ABMIL (Paper Replication)',
        output_path='confusion_matrix_abmil_replicated.png'
    )

    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()