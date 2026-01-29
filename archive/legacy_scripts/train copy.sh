#!/bin/bash
# MIL Model Training

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths
LABELS_CSV="panda_labels.csv"
FEATURES_DIR="/media/nadim/Data/Imaging/prostate-cancer-grade-assessment/trident_processedqc/20x_256px_0px_overlap/features_uni_v2/"

# Model settings
MODEL="abmil.base.uni_v2.pc108-24k"
NUM_CLASSES=6

# Training settings
EPOCHS=20
EARLY_STOPPING_PATIENCE=100
MIN_EPOCHS=10
OUTPUT_DIR="experiments"

# =============================================================================
# RUN
# =============================================================================

python train_mil.py \
    --labels-csv "$LABELS_CSV" \
    --features-dir "$FEATURES_DIR" \
    --model "$MODEL" \
    --num-classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --min-epochs "$MIN_EPOCHS" \
    --split-column split \
    --output-dir "$OUTPUT_DIR"
