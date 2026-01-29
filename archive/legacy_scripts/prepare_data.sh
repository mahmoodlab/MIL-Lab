#!/bin/bash
# Data Preparation for MIL Training
# Converts PathBench TSV to labels CSV format

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
TSV_PATH="/media/nadim/Data/Imaging/prostate-cancer-grade-assessment/k=all.tsv"
FEATURES_DIR="/media/nadim/Data/Imaging/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/"
OUTPUT_CSV="panda_labels.csv"

# Data settings
FOLD_COLUMN="fold_0"
VAL_FRACTION=0.1
MIN_PATCHES=24
SEED=10

# =============================================================================
# RUN
# =============================================================================

python prepare_panda_data.py \
    --tsv "$TSV_PATH" \
    --features "$FEATURES_DIR" \
    --output "$OUTPUT_CSV" \
    --fold-column "$FOLD_COLUMN" \
    --val-fraction "$VAL_FRACTION" \
    --min-patches "$MIN_PATCHES" \
    --seed "$SEED"
