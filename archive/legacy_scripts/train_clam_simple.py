#!/usr/bin/env python3
"""
Simplified CLAM training script using shared utilities
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
WSI_DIR = '/media/nadim/Data/prostate-cancer-grade-assessment/train_images'

# Data settings
GRADE_GROUP = True
EXCLUDE_MID_GRADE = True
SEED = 10
TRAIN_SEED = 42

# Training settings
NUM_EPOCHS = 2
BATCH_SIZE = 1  # CLAM requires batch_size=1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
K_SAMPLE = 8  # Sample 8 top/bottom patches for instance loss
BAG_WEIGHT = 0.7  # 70% bag loss, 30% instance loss

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
        num_features=None,  # Use all patches for CLAM
        num_workers=4,
        seed=TRAIN_SEED
    )

    print(f"Train samples: {len(train_loader)}")
    print(f"Val samples: {len(val_loader)}")
    print(f"Test samples: {len(test_loader)}\n")

    # Step 3: Create model
    print("="*70)
    print("LOADING CLAM MODEL")
    print("="*70 + "\n")

    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        'clam.base_ce.uni_v2.none',
        num_classes=num_classes,
        k_sample=K_SAMPLE,
        bag_weight=BAG_WEIGHT
    ).to(device)

    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Step 4: Training
    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        total_loss = 0.
        total_cls_loss = 0.
        total_inst_loss = 0.

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
            loss = results_dict['loss']

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += log_dict['cls_loss']
            total_inst_loss += log_dict['instance_loss']

            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cls': f"{log_dict['cls_loss']:.4f}",
                'inst': f"{log_dict['instance_loss']:.4f}"
            })

        avg_train_loss = total_loss / len(train_loader)
        avg_cls_loss = total_cls_loss / len(train_loader)
        avg_inst_loss = total_inst_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]'):
                features, labels = features.to(device), labels.to(device)

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

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Cls: {avg_cls_loss:.4f}, Inst: {avg_inst_loss:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        print(f"  LR:         {new_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_clam.pth')
            print(f"  >>> Saved best model")

        print("-"*70)

    # Step 5: Evaluation
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70 + "\n")

    model.load_state_dict(torch.load('best_model_clam.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Testing'):
            features, labels = features.to(device), labels.to(device)
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
    print(f"  Quadratic Kappa:  {test_kappa:.4f}\n")

    # Save confusion matrix
    plot_labels = [label.replace(' ', '\n') for label in class_labels]
    plot_confusion_matrix(
        all_labels, all_preds, plot_labels,
        title='Confusion Matrix - CLAM',
        output_path='confusion_matrix_clam.png'
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
