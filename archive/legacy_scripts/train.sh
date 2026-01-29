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
NUM_HEADS=1  # Number of attention heads (ABMIL)

# Training settings
EPOCHS=20
EARLY_STOPPING_PATIENCE=100
MIN_EPOCHS=10
OUTPUT_DIR="experiments"
TASK_TYPE="multiclass"  # "binary" or "multiclass"
EARLY_STOPPING_METRIC="kappa"  # "auto", "kappa", "balanced_accuracy", "auc"

# =============================================================================
# RUN
# =============================================================================

python train_mil.py \
    --labels-csv "$LABELS_CSV" \
    --features-dir "$FEATURES_DIR" \
    --model "$MODEL" \
    --num-classes "$NUM_CLASSES" \
    --num-heads "$NUM_HEADS" \
    --epochs "$EPOCHS" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --min-epochs "$MIN_EPOCHS" \
    --task-type "$TASK_TYPE" \
    --early-stopping-metric "$EARLY_STOPPING_METRIC" \
    --split-column split \
    --output-dir "$OUTPUT_DIR"
