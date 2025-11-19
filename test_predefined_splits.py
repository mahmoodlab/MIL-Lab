#!/usr/bin/env python3
"""
Test script to verify the predefined splits loading functionality
"""

from utils import load_panda_predefined_splits

# Test with k=all.tsv
TSV_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/k=all.tsv'
FEATS_PATH = '/media/nadim/Data/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/'

print("Testing predefined splits loading...\n")

# Test 1: Load with 10% validation split
print("="*70)
print("TEST 1: Load with 10% validation split from training data")
print("="*70)
df, num_classes, class_labels = load_panda_predefined_splits(
    TSV_PATH, FEATS_PATH,
    fold_column='fold_0',
    val_fraction=0.1,
    grade_group=False,
    exclude_mid_grade=False,
    seed=10
)

print("\nDataFrame head:")
print(df.head())

print("\nClass distribution by split:")
print(df.groupby(['split', 'label']).size())

print("\n" + "="*70)
print("TEST 1 PASSED ✓")
print("="*70 + "\n")
