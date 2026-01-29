#!/usr/bin/env python3
"""
Compare labels between train.csv and k=all.tsv
Check if 'label' column in train.csv matches 'isup_grade' in k=all.tsv
"""

import pandas as pd

# Read both files
print("Reading files...")
train_csv_path = '/media/nadim/Data/prostate-cancer-grade-assessment/train.csv'
k_all_tsv_path = '/media/nadim/Data/prostate-cancer-grade-assessment/k=all.tsv'

df_train = pd.read_csv(train_csv_path)
df_k_all = pd.read_csv(k_all_tsv_path, sep='\t')

print(f"train.csv: {len(df_train)} rows")
print(f"k=all.tsv: {len(df_k_all)} rows\n")

# Merge on slide_id
print("Merging on slide_id...")
df_merged = df_train.merge(df_k_all, on='slide_id', how='inner')
print(f"Common slides: {len(df_merged)}\n")

# Compare labels
print("Comparing 'label' (train.csv) vs 'isup_grade' (k=all.tsv)...")
df_merged['match'] = df_merged['label'] == df_merged['isup_grade']

# Calculate statistics
num_matches = df_merged['match'].sum()
num_mismatches = (~df_merged['match']).sum()
total = len(df_merged)
match_rate = (num_matches / total * 100) if total > 0 else 0

print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)
print(f"Total slides compared:  {total}")
print(f"Matches:                {num_matches} ({match_rate:.2f}%)")
print(f"Mismatches:             {num_mismatches} ({100-match_rate:.2f}%)")
print("="*70 + "\n")

# Show mismatches if any
if num_mismatches > 0:
    print("MISMATCHES:")
    print("-"*70)
    mismatches = df_merged[~df_merged['match']][['slide_id', 'label', 'isup_grade']]
    print(mismatches.to_string(index=False))
    print("\n")

    # Show distribution of mismatches
    print("Mismatch Summary by Label:")
    print("-"*70)
    for label in sorted(df_merged['label'].unique()):
        subset = df_merged[df_merged['label'] == label]
        label_total = len(subset)
        label_mismatches = (~subset['match']).sum()
        if label_mismatches > 0:
            print(f"Label {label}: {label_mismatches}/{label_total} mismatches")
else:
    print("✓ All labels match perfectly!")

# Check for slides in train.csv but not in k=all.tsv
print("\n" + "="*70)
print("COVERAGE CHECK")
print("="*70)
slides_only_in_train = set(df_train['slide_id']) - set(df_k_all['slide_id'])
slides_only_in_k_all = set(df_k_all['slide_id']) - set(df_train['slide_id'])

print(f"Slides only in train.csv:  {len(slides_only_in_train)}")
print(f"Slides only in k=all.tsv:  {len(slides_only_in_k_all)}")
print("="*70)
