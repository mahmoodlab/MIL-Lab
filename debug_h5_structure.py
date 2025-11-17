#!/usr/bin/env python3
"""
Diagnostic script to check h5 file structure and trace tensor shapes
"""

import h5py
import torch
import numpy as np
import os
from glob import glob

# Paths from the notebook
feats_path = '/media/nadim/Data/prostate-cancer-grade-assessment/panda/'

print("="*70)
print("H5 FILE STRUCTURE DIAGNOSTIC")
print("="*70)

# Find a few h5 files
h5_files = glob(os.path.join(feats_path, '*.h5'))[:5]

if not h5_files:
    print(f"ERROR: No h5 files found in {feats_path}")
    exit(1)

print(f"\nFound {len(glob(os.path.join(feats_path, '*.h5')))} total h5 files")
print(f"Checking first 5 files...\n")

for i, h5_file in enumerate(h5_files, 1):
    print(f"\n{i}. File: {os.path.basename(h5_file)}")
    print("-" * 70)

    try:
        with h5py.File(h5_file, 'r') as f:
            print(f"   Keys in file: {list(f.keys())}")

            # Check features dataset
            if 'features' in f:
                features_shape = f['features'].shape
                features_dtype = f['features'].dtype
                print(f"   'features' shape: {features_shape}")
                print(f"   'features' dtype: {features_dtype}")

                # Test the current loading code
                print("\n   Testing current loading code:")
                print(f"   >>> features = torch.from_numpy(f['features'][:]).squeeze(0)")

                # Step by step
                raw_features = f['features'][:]
                print(f"       After f['features'][:] -> shape: {raw_features.shape}")

                tensor_features = torch.from_numpy(raw_features)
                print(f"       After torch.from_numpy() -> shape: {tensor_features.shape}")

                squeezed_features = tensor_features.squeeze(0)
                print(f"       After .squeeze(0) -> shape: {squeezed_features.shape}")

                # Expected shape
                print(f"\n   Expected shape after squeeze(0): (num_patches, 1536)")
                if len(squeezed_features.shape) == 2 and squeezed_features.shape[1] == 1536:
                    print(f"   ✓ Shape looks CORRECT: {squeezed_features.shape}")
                else:
                    print(f"   ✗ Shape looks WRONG: {squeezed_features.shape}")
                    print(f"   ⚠ This will cause issues!")

                # Test what happens with batch_size=1
                print(f"\n   Simulating DataLoader with batch_size=1:")
                batch = [squeezed_features]
                try:
                    stacked = torch.stack(batch)
                    print(f"       After torch.stack([features]) -> shape: {stacked.shape}")
                    print(f"       Expected: (1, num_patches, 1536)")
                    if len(stacked.shape) == 3:
                        print(f"       ✓ Batching works correctly")
                    else:
                        print(f"       ✗ Batching produces wrong shape!")
                except Exception as e:
                    print(f"       ✗ ERROR during batching: {e}")

            else:
                print("   ✗ 'features' key not found!")

            # Check coords if present
            if 'coords_patching' in f:
                coords_shape = f['coords_patching'].shape
                print(f"\n   'coords_patching' shape: {coords_shape}")

    except Exception as e:
        print(f"   ✗ ERROR reading file: {e}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("""
Based on the output above:

1. If 'features' shape is (1, num_patches, 1536):
   ✓ Your h5 files are in the CORRECT format
   ✓ The .squeeze(0) correctly gives (num_patches, 1536)
   ✓ DataLoader batching should work

2. If 'features' shape is (num_patches, 1536):
   ✗ Your h5 files are missing the leading dimension
   ✗ The .squeeze(0) might remove the wrong dimension
   → SOLUTION: Remove .squeeze(0) or add unsqueeze

3. If shapes are inconsistent or unexpected:
   ✗ Your embedding generation might have issues
   → SOLUTION: Regenerate embeddings with correct format
""")

print("\nTo fix ABMIL-Feather notebook based on findings:")
print("Run this script and share the output!")
