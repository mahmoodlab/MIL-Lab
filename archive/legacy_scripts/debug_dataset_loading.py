#!/usr/bin/env python3
"""
Test the actual dataset loading and model forward pass
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob

# Configuration (from notebook)
csv_path = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
feats_path = '/media/nadim/Data/prostate-cancer-grade-assessment/panda/'
SEED = 42

# Custom dataset (from notebook)
class PANDAH5Dataset(Dataset):
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
            # UNI v2 features have shape (1, num_patches, 1536)
            features = torch.from_numpy(f["features"][:]).squeeze(0).clone()  # (num_patches, 1536)

        # Sample patches for training to control memory
        if self.split == 'train':
            num_available = features.shape[0]
            if num_available >= self.num_features:
                indices = torch.randperm(num_available, generator=torch.Generator().manual_seed(SEED))[:self.num_features]
            else:
                indices = torch.randint(num_available, (self.num_features,), generator=torch.Generator().manual_seed(SEED))
            features = features[indices]

        label = torch.tensor(row["label"], dtype=torch.long)

        print(f"    __getitem__({idx}): features.shape = {features.shape}, label = {label.item()}")
        return features, label

print("="*70)
print("DATASET LOADING TEST")
print("="*70)

# Quick data prep
print("\n1. Loading CSV and creating small test split...")
df_labels = pd.read_csv(csv_path)[['slide_id', 'label']]
feature_files = glob(os.path.join(feats_path, '*.h5'))
available_ids = [os.path.basename(f).replace('.h5', '') for f in feature_files]
df_matched = df_labels[df_labels['slide_id'].isin(available_ids)].reset_index(drop=True)

# Take just 10 samples for testing
df_test = df_matched.head(10).copy()
df_test['split'] = 'val'

print(f"   Created test set with {len(df_test)} samples")

# Create dataset
print("\n2. Creating PANDAH5Dataset...")
val_dataset = PANDAH5Dataset(feats_path, df_test, "val", num_features=512)
print(f"   Dataset length: {len(val_dataset)}")

# Test __getitem__
print("\n3. Testing __getitem__ on first 3 samples:")
for i in range(min(3, len(val_dataset))):
    features, label = val_dataset[i]
    print(f"   Sample {i}: shape={features.shape}, dtype={features.dtype}, label={label}")

# Create DataLoader
print("\n4. Creating DataLoader with batch_size=1...")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"   DataLoader created")

# Test DataLoader iteration
print("\n5. Testing DataLoader iteration (first 3 batches):")
for i, (features_batch, labels_batch) in enumerate(val_loader):
    print(f"   Batch {i}: features.shape={features_batch.shape}, labels.shape={labels_batch.shape}")
    print(f"            features.dtype={features_batch.dtype}, requires_grad={features_batch.requires_grad}")

    if i >= 2:
        break

# Test with batch_size=32
print("\n6. Testing DataLoader with batch_size=32...")
try:
    train_dataset = PANDAH5Dataset(feats_path, df_test, "val", num_features=512)  # Use val but test batching
    # Change split to train temporarily to enable sampling
    train_dataset.split = 'train'

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)  # batch_size=2 for quick test

    print("   Attempting to load first batch...")
    features_batch, labels_batch = next(iter(train_loader))
    print(f"   ✓ Success! features.shape={features_batch.shape}, labels.shape={labels_batch.shape}")
except Exception as e:
    print(f"   ✗ ERROR: {type(e).__name__}: {e}")

# Test model forward pass
print("\n7. Testing model forward pass...")
try:
    from src.builder import create_model

    print("   Creating ABMIL model...")
    model = create_model(
        'abmil.base.uni_v2.pc108-24k',
        num_classes=6,
        dropout=0.2,
        gate=True
    )
    model.eval()

    print("   Model created successfully")

    # Test with a single sample
    print("\n   Testing forward pass with batch_size=1...")
    features_batch, labels_batch = next(iter(val_loader))
    print(f"   Input shape: {features_batch.shape}")

    with torch.no_grad():
        results_dict, log_dict = model(features_batch)
        logits = results_dict['logits']
        print(f"   ✓ Forward pass successful!")
        print(f"   Output logits shape: {logits.shape}")

except Exception as e:
    print(f"   ✗ ERROR during model forward: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
If you see errors above:
1. Check the exact shape where the error occurs
2. The error 'batch1 must be a 3D tensor' means the model is receiving wrong input shape
3. My .clone() fix is still needed for batch_size > 1
""")
