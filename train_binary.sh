#!/bin/bash
# MIL Model Training - Binary Classification with Multi-Slide Cases

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths (UPDATE THESE)
LABELS_CSV="/path/to/your/labels.csv"
FEATURES_DIR="/path/to/your/features/"

# Model settings
MODEL="abmil.base.uni"          # Using UNI v1 (1024 dim), no pretrained weights
NUM_CLASSES=2                    # Binary classification

# Training settings
EPOCHS=50
EARLY_STOPPING_PATIENCE=10
MIN_EPOCHS=10
OUTPUT_DIR="experiments_binary"

# Multi-slide grouping settings
GROUP_COLUMN="case_id"           # Column to group slides by
FUSION="early"                   # early = concatenate slides into Giant Bag

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
    --group-column "$GROUP_COLUMN" \
    --fusion "$FUSION" \
    --output-dir "$OUTPUT_DIR"
