#!/usr/bin/env python3
"""
Standalone script to train ABMIL FEATHER-24K on PANDA dataset
Converted from notebook to rule out Jupyter-specific issues
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob

# Import MIL-Lab model builder
from src.builder import create_model

# ============================================================================
# CONFIGURATION
# ============================================================================
csv_path = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
feats_path = feats_path = '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/'

SEED = 10
TRAIN_SEED = 42

# Grade grouping flags
grade_group = True
exclude_mid_grade = True

# Training hyperparameters
num_epochs = 3
batch_size = 1  # Changed from 32 to 1 for standard MIL
learning_rate = 1e-4
weight_decay = 1e-5
num_features = 512  # Number of patches to sample for training

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================
print("="*70)
print("IMPROVED DATA PREPROCESSING")
if grade_group:
    if exclude_mid_grade:
        print("MODE: Clinical Grade Grouping (3 classes, mid grade excluded)")
    else:
        print("MODE: Clinical Grade Grouping (4 classes)")
else:
    print("MODE: Original ISUP Grading (6 classes)")
print("="*70 + "\n")

# Set random seed for reproducibility
np.random.seed(SEED)

# Step 1: Read labels and slide IDs from CSV
print("Step 1: Reading labels from CSV...")
df_labels = pd.read_csv(csv_path)[['slide_id', 'label']]
df_labels['isup_grade'] = df_labels['label']

# Apply grade grouping if enabled
if grade_group:
    if exclude_mid_grade:
        print(f"  Excluding mid grade (ISUP 2-3) from analysis...")
        df_labels = df_labels[~df_labels['isup_grade'].isin([2, 3])].reset_index(drop=True)
        print(f"  Remaining slides after exclusion: {len(df_labels)}")

    def map_isup_to_group(isup_grade):
        """Map ISUP grades to clinical groups"""
        if isup_grade == 0:
            return 0  # No cancer
        elif isup_grade == 1:
            return 1  # Low grade
        elif isup_grade in [2, 3]:
            return 2  # Mid grade
        elif isup_grade in [4, 5]:
            return 2 if exclude_mid_grade else 3  # High grade
        else:
            raise ValueError(f"Invalid ISUP grade: {isup_grade}")

    df_labels['label'] = df_labels['isup_grade'].apply(map_isup_to_group)
    print(f"  Applied grade grouping: ISUP -> Clinical Groups")

print(f"  Found {len(df_labels)} slides in CSV with labels")

if grade_group:
    if exclude_mid_grade:
        group_names = ['Group 0 (No cancer)', 'Group 1 (Low grade)', 'Group 2 (High grade)']
    else:
        group_names = ['Group 0 (No cancer)', 'Group 1 (Low grade)',
                       'Group 2 (Mid grade)', 'Group 3 (High grade)']
else:
    group_names = [f'ISUP {i}' for i in range(6)]

# Step 2: Find all available feature files
print(f"\nStep 2: Scanning features directory...")
feature_files = glob(os.path.join(feats_path, '*.h5'))
available_slide_ids = [os.path.basename(f).replace('.h5', '') for f in feature_files]
print(f"  Found {len(available_slide_ids)} feature files")

# Step 3: Match CSV with available features
print(f"\nStep 3: Matching CSV labels with available features...")
df_labels['has_features'] = df_labels['slide_id'].isin(available_slide_ids)
df_matched = df_labels[df_labels['has_features']].drop(columns=['has_features']).reset_index(drop=True)
missing_count = len(df_labels) - len(df_matched)
print(f"  Matched: {len(df_matched)} slides")
print(f"  Missing features: {missing_count} slides")

# Step 4: Perform stratified train/val/test split
print(f"\nStep 4: Performing stratified split (70% train, 20% val, 10% test)...")
train_val_df, test_df = train_test_split(
    df_matched, test_size=0.10, stratify=df_matched['label'], random_state=SEED
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.222, stratify=train_val_df['label'], random_state=SEED
)

train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"\n{'='*70}")
print("SPLIT SUMMARY")
print(f"{'='*70}")
print(f"Total slides: {len(df)}\n")
print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

num_classes = len(df['label'].unique())
print(f"\nNumber of classes: {num_classes}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 2: DATASET AND MODEL
# ============================================================================

class PANDAH5Dataset(Dataset):
    """Custom dataset for PANDA with UNI v2 features"""
    def __init__(self, feats_path, df, split, num_features=512):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.feats_path = feats_path
        self.num_features = num_features
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = os.path.join(self.feats_path, row['slide_id'] + '.h5')

        with h5py.File(feat_path, "r") as f:
            # Load features and handle both 2D (new Trident) and 3D (old) formats
            raw_features = torch.from_numpy(f["features"][:])

            # Trident generates 2D: (num_patches, 1536)
            # Old format was 3D: (1, num_patches, 1536)
            if len(raw_features.shape) == 3:
                # Old format - squeeze first dimension
                features = raw_features.squeeze(0).clone()
            else:
                # New Trident format - already 2D
                features = raw_features.clone()

        # Sample patches for training to control memory
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(TRAIN_SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(TRAIN_SEED))
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


print("="*70)
print("LOADING PRETRAINED ABMIL FEATHER-24K MODEL")
print("="*70 + "\n")

# Set deterministic behavior
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)
torch.cuda.manual_seed_all(TRAIN_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Create ABMIL model with pretrained weights
print("Creating model with pretrained weights...")
model = create_model(
    'abmil.base.uni_v2.pc108-24k',  # Pretrained FEATHER-24K model
    num_classes=num_classes,
    dropout=0.25,  # Changed from 0.2 to 0.25
    gate=True
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Create datasets
train_dataset = PANDAH5Dataset(feats_path, df, "train", num_features=num_features)
val_dataset = PANDAH5Dataset(feats_path, df, "val", num_features=num_features)
test_dataset = PANDAH5Dataset(feats_path, df, "test", num_features=num_features)

# Create weighted sampler for class balancing
train_df = df[df['split'] == 'train']
train_labels = train_df['label'].values
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

print(f"\nClass distribution in training set:")
for i, count in enumerate(class_counts):
    print(f"  Class {i}: {count} samples (weight: {class_weights[i]:.4f})")

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create dataloaders with weighted sampling
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

print(f"\nDataset sizes:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val samples: {len(val_loader)}")
print(f"  Test samples: {len(test_loader)}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 3: TRAINING
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
train_losses = []
val_losses = []
val_accuracies = []

print("\n" + "="*70)
print("Starting ABMIL fine-tuning...")
print("="*70 + "\n")

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    for batch_idx, (features, labels) in enumerate(train_bar):
        features, labels = features.to(device), labels.to(device)

        # DEBUG: Print shapes for first batch
        if epoch == 0 and batch_idx == 0:
            print(f"\n[DEBUG] First batch:")
            print(f"  features.shape = {features.shape}")
            print(f"  features.dtype = {features.dtype}")
            print(f"  labels.shape = {labels.shape}")
            print(f"  features.requires_grad = {features.requires_grad}")
            print(f"  features.is_contiguous() = {features.is_contiguous()}\n")

        optimizer.zero_grad()

        # ABMIL forward
        try:
            results_dict, log_dict = model(features, loss_fn=criterion, label=labels)
            loss = results_dict['loss']
        except Exception as e:
            print(f"\n[ERROR] Exception during forward pass:")
            print(f"  Batch index: {batch_idx}")
            print(f"  features.shape: {features.shape}")
            print(f"  labels.shape: {labels.shape}")
            print(f"  Exception: {type(e).__name__}: {e}")
            raise

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

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

    # Learning rate scheduling
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val_loss)
    new_lr = optimizer.param_groups[0]['lr']

    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Train Loss:     {avg_train_loss:.4f}")
    print(f"  Val Loss:       {avg_val_loss:.4f}")
    print(f"  Val Acc:        {val_acc:.4f}")
    print(f"  LR:             {new_lr:.6f}")

    if new_lr < old_lr:
        print(f"  >>> Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model_abmil_feather_panda.pth')
        print(f"  >>> Saved best model (Val Loss: {best_val_loss:.4f})")

    print("-"*70)

print("\n" + "="*70)
print("Training complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("="*70 + "\n")

# ============================================================================
# STEP 4: EVALUATION
# ============================================================================

print("Loading best model for evaluation...")
model.load_state_dict(torch.load('best_model_abmil_feather_panda.pth'))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for features, labels in tqdm(test_loader, desc='Testing'):
        features, labels = features.to(device), labels.to(device)

        results_dict, log_dict = model(features)
        logits = results_dict['logits']

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
test_acc = accuracy_score(all_labels, all_preds)
test_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
test_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

print(f"\n{'='*70}")
print("ABMIL FEATHER-24K Test Results")
print(f"{'='*70}")
print(f"Test Accuracy:                    {test_acc:.4f}")
print(f"Test Balanced Accuracy:           {test_balanced_acc:.4f}")
print(f"Test Quadratic Weighted Kappa:    {test_kappa:.4f}")
print(f"{'='*70}\n")

# Save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))

if grade_group:
    if exclude_mid_grade:
        class_labels = ['Group 0\n(No cancer)', 'Group 1\n(Low grade)', 'Group 2\n(High grade)']
    else:
        class_labels = ['Group 0\n(No cancer)', 'Group 1\n(Low grade)',
                        'Group 2\n(Mid grade)', 'Group 3\n(High grade)']
else:
    class_labels = [f'ISUP {i}' for i in range(num_classes)]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - PANDA Test Set (ABMIL FEATHER-24K)')
plt.savefig('confusion_matrix_abmil_feather_panda.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved to: confusion_matrix_abmil_feather_panda.png")

print("\nTraining script completed successfully!")
